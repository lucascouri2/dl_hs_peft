"""
Sistema de Classificação Zero-Shot e Few-Shot
"""

import torch
import numpy as np
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from datasets import Dataset
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)


@dataclass
class ZeroFewShotConfig:
    """Configurações para zero-shot e few-shot classification"""
    
    model_name: str
    cache_dir: str = "./cache"
    
    # Quantização
    use_4bit: bool = True
    
    # Few-shot
    num_shots: int = 0  # 0 = zero-shot, >0 = few-shot
    
    # Geração
    max_new_tokens: int = 10
    temperature: float = 0.1
    do_sample: bool = False
    
    # Processamento
    batch_size: int = 8
    max_length: int = 2048
    
    # Outros
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ZeroFewShotClassifier:
    """
    Classificador Zero-Shot e Few-Shot
    Você passa o prompt template como argumento
    """
    
    def __init__(self, config: ZeroFewShotConfig, label_names: List[str]):
        """
        Args:
            config: Configurações
            label_names: Lista de nomes das classes (ex: ["NON-HATE", "HATE"])
        """
        self.config = config
        self.label_names = label_names
        self.num_classes = len(label_names)
        
        # Carregar modelo e tokenizer
        self.model, self.tokenizer = self._load_model()
        
        print(f"\nModelo carregado: {config.model_name}")
        print(f"Modo: {'Zero-Shot' if config.num_shots == 0 else f'{config.num_shots}-Shot'}")
        print(f"Classes: {label_names}")
    
    def _load_model(self):
        """Carrega modelo e tokenizer"""
        
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                cache_dir=self.config.cache_dir,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.config.device)
        
        # Configurar pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        
        return model, tokenizer
    
    def _format_prompt(self, text: str, prompt_template: str, 
                       few_shot_examples: Optional[List[Dict]] = None) -> str:
        """
        Formata o prompt final
        
        Args:
            text: Texto a ser classificado
            prompt_template: Template do prompt com placeholder {text} e opcionalmente {examples}
            few_shot_examples: Lista de dicionários com 'text' e 'label'
        
        Returns:
            Prompt formatado
        """
        # Se for few-shot e tiver exemplos
        if few_shot_examples and self.config.num_shots > 0:
            examples_str = ""
            for i, ex in enumerate(few_shot_examples, 1):
                label_name = self.label_names[ex['label']]
                examples_str += f"\nExemplo {i}:\nTexto: {ex['text']}\nClasse: {ex['label']} ({label_name})\n"
            
            # Substituir placeholders
            prompt = prompt_template.replace("{examples}", examples_str)
            prompt = prompt.replace("{text}", text)
        else:
            # Zero-shot: apenas substituir texto
            prompt = prompt_template.replace("{text}", text)
            # Remover placeholder de examples se existir
            prompt = prompt.replace("{examples}", "")
        
        return prompt
    
    def _select_few_shot_examples(self, train_dataset: Dataset) -> List[Dict]:
        """
        Seleciona exemplos para few-shot (balanceado por classe)
        
        Args:
            train_dataset: Dataset de treino
            
        Returns:
            Lista de exemplos balanceados
        """
        df = pd.DataFrame(train_dataset)
        examples = []
        
        # Selecionar num_shots exemplos de cada classe
        for label in range(self.num_classes):
            class_samples = df[df['label'] == label]
            n_samples = min(self.config.num_shots, len(class_samples))
            
            samples = class_samples.sample(n=n_samples, random_state=self.config.seed)
            
            for _, row in samples.iterrows():
                examples.append({
                    'text': row['text'],
                    'label': row['label']
                })
        
        return examples
    
    def _generate_prediction(self, prompt: str) -> int:
        """
        Gera predição para um prompt
        
        Args:
            prompt: Prompt formatado
            
        Returns:
            Label predito (0, 1, 2, ...)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodificar apenas os novos tokens gerados
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Extrair label da resposta
        #prediction = self._extract_label(generated_text)
        prediction = self._extract_label_improved(generated_text, self.num_classes, self.label_names)
        
        return prediction
    
    def _extract_label(self, text: str) -> int:
        """
        Extrai label numérico da resposta gerada
        
        Args:
            text: Texto gerado pelo modelo
            
        Returns:
            Label numérico (0, 1, 2, ...) ou 0 se não conseguir extrair
        """        
        # Tentar extrair primeiro número
        numbers = re.findall(r'\d+', text)
        
        if numbers:
            label = int(numbers[0])
            if 0 <= label < self.num_classes:
                return label
        
        # Fallback: retornar classe 0
        return 0
    
    def _extract_label_improved(self, text: str, num_classes: int, label_names: list = None) -> int:
        """
        Extração de label melhorada com múltiplas estratégias
        
        Args:
            text: Texto gerado pelo modelo
            num_classes: Número de classes
            label_names: Nomes das classes (opcional)
        
        Returns:
            Label extraído
        """
        text_lower = text.lower().strip()
        
        # Estratégia 1: Primeiro dígito isolado
        # Procura por dígito seguido de espaço, fim de linha ou pontuação
        pattern = r'\b(\d+)\b'
        matches = re.findall(pattern, text)
        if matches:
            label = int(matches[0])
            if 0 <= label < num_classes:
                return label
        
        # Estratégia 2: Apenas primeiro dígito qualquer
        numbers = re.findall(r'\d+', text)
        if numbers:
            label = int(numbers[0])
            if 0 <= label < num_classes:
                return label
        
        # Estratégia 3: Buscar por palavras-chave (para binário)
        if num_classes == 2:
            # Positivos (classe 1 - HATE)
            hate_keywords = ['ódio', 'hate', 'sim', 'yes', 'contém', 'tem', 'possui', 
                            'verdadeiro', 'true', 'positivo', 'positive']
            if any(kw in text_lower for kw in hate_keywords):
                # Verificar se não tem negação antes
                no_keywords = ['não', 'nao', 'sem', 'no ', 'not', 'isn\'t', 'falso', 'false']
                if not any(kw in text_lower for kw in no_keywords):
                    return 1
            
            # Negativos (classe 0 - NON-HATE)
            non_hate_keywords = ['não', 'nao', 'sem', 'no ', 'not', 'isn\'t', 
                                'falso', 'false', 'negativo', 'negative']
            if any(kw in text_lower for kw in non_hate_keywords):
                return 0
        
        # Estratégia 4: Buscar pelos nomes das classes
        if label_names:
            for i, name in enumerate(label_names):
                name_lower = name.lower()
                if name_lower in text_lower:
                    return i
        
        # Estratégia 5: Se texto muito curto e tem apenas número
        if len(text.strip()) <= 3:
            try:
                label = int(text.strip())
                if 0 <= label < num_classes:
                    return label
            except:
                pass
        
        # Fallback: retornar classe 0
        print(f"Não foi possível extrair label de: '{text}' - usando 0")
        return 0
    
    def predict(self, test_dataset: Dataset, 
                prompt_template: str,
                train_dataset: Optional[Dataset] = None,
                verbose: bool = True) -> np.ndarray:
        """
        Realiza predições
        
        Args:
            test_dataset: Dataset de teste com colunas 'text' e 'label'
            prompt_template: Template do prompt com {text} e opcionalmente {examples}
            train_dataset: Dataset de treino (necessário para few-shot)
            verbose: Se True, mostra progresso
            
        Returns:
            Array numpy com predições
        """
        predictions = []
        
        # Preparar exemplos para few-shot se necessário
        few_shot_examples = None
        if self.config.num_shots > 0:
            if train_dataset is None:
                raise ValueError("train_dataset é necessário para few-shot learning")
            
            few_shot_examples = self._select_few_shot_examples(train_dataset)
            
            if verbose:
                print(f"\nSelecionados {len(few_shot_examples)} exemplos para few-shot")
        
        # Processar em batches
        texts = test_dataset['text']
        total = len(texts)
        
        if verbose:
            print(f"\nProcessando {total} exemplos...")
        
        for i in range(0, total, self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            for text in batch_texts:
                # Formatar prompt
                prompt = self._format_prompt(text, prompt_template, few_shot_examples)
                
                # Gerar predição
                pred = self._generate_prediction(prompt)
                predictions.append(pred)
            
            # Progress
            if verbose and (i // self.config.batch_size + 1) % 10 == 0:
                processed = min(i + self.config.batch_size, total)
                print(f"Processados {processed}/{total} exemplos")
        
        if verbose:
            print(f"✓ Predições concluídas!")
        
        return np.array(predictions)
    
    def evaluate(self, test_dataset: Dataset,
             prompt_template: str,
             train_dataset: Optional[Dataset] = None,
             verbose: bool = True,
             save_plots: bool = True,
             output_dir: Optional[str] = None) -> Dict:
        """
        Avalia o modelo com métricas completas (igual ao ModelEvaluator)
        
        Args:
            test_dataset: Dataset de teste
            prompt_template: Template do prompt
            train_dataset: Dataset de treino (para few-shot)
            verbose: Se True, mostra resultados
            save_plots: Se True, salva gráficos
            output_dir: Diretório para salvar (None = não salva)
            
        Returns:
            Dicionário com métricas
        """
        
        
        # Obter predições e probabilidades
        predictions = self.predict(test_dataset, prompt_template, train_dataset, verbose)
        true_labels = np.array(test_dataset['label'])
        
        # OBTER PROBABILIDADES (reprocessar para ter softmax)       
        print("\nCalculando probabilidades..." if verbose else "")
        probabilities = []
        
        # Preparar exemplos para few-shot se necessário
        few_shot_examples = None
        if self.config.num_shots > 0:
            if train_dataset is None:
                raise ValueError("train_dataset é necessário para few-shot learning")
            few_shot_examples = self._select_few_shot_examples(train_dataset)
        
        texts = test_dataset['text']
        
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            for text in batch_texts:
                prompt = self._format_prompt(text, prompt_template, few_shot_examples)
                
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length
                ).to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    
                    # Softmax para probabilidades
                    probs = torch.softmax(logits[:, -1, :], dim=-1)[:, :self.num_classes]
                    probabilities.append(probs.cpu().numpy()[0])
        
        probabilities = np.array(probabilities)
        
        # CALCULAR MÉTRICAS        
        # Básicas
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_per_class = f1_score(true_labels, predictions, average=None)
        accuracy = accuracy_score(true_labels, predictions)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Average Precision por classe
        avg_precision_scores = {}
        for i, label_name in enumerate(self.label_names):
            y_true_binary = (true_labels == i).astype(int)
            y_scores = probabilities[:, i]
            avg_precision_scores[label_name] = average_precision_score(y_true_binary, y_scores)
        
        macro_avg_precision = np.mean(list(avg_precision_scores.values()))
        
        # ROC-AUC
        if self.num_classes == 2:
            roc_auc = roc_auc_score(true_labels, probabilities[:, 1])
        else:
            roc_auc = roc_auc_score(true_labels, probabilities, 
                                multi_class='ovr', average='macro')
        
        # ECE (Expected Calibration Error)
        ece = compute_ece(true_labels, probabilities, n_bins=10)
        
        # Classification report
        clf_report = classification_report(
            true_labels, predictions,
            target_names=self.label_names,
            output_dict=True,
            zero_division=0
        )
        
        # MONTAR RESULTADOS        
        results = {
            # Principais
            'macro_f1': float(f1_macro),
            'accuracy': float(accuracy),
            'f1_per_class': {
                self.label_names[i]: float(f1) 
                for i, f1 in enumerate(f1_per_class)
            },
            
            # Para desbalanceamento
            'macro_avg_precision': float(macro_avg_precision),
            'avg_precision_per_class': {
                k: float(v) for k, v in avg_precision_scores.items()
            },
            'roc_auc': float(roc_auc),
            
            # Calibração
            'expected_calibration_error': float(ece),
            
            # Detalhes
            'precision_per_class': {
                label: float(clf_report[label]['precision'])
                for label in self.label_names
            },
            'recall_per_class': {
                label: float(clf_report[label]['recall'])
                for label in self.label_names
            },
            
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': clf_report,
            'predictions': predictions.tolist(),
            'true_labels': true_labels.tolist()
        }
        
        # IMPRIMIR        
        if verbose:
            print(f"\n{'='*50}")
            print("RESULTADOS")
            print(f"{'='*50}")
            print(f"Macro-F1: {f1_macro:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Macro Avg Precision: {macro_avg_precision:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"Expected Calibration Error: {ece:.4f}")
            
            print(f"\n{'='*50}")
            print("POR CLASSE")
            print(f"{'='*50}")
            for i, label in enumerate(self.label_names):
                print(f"\n{label}:")
                print(f"  F1: {f1_per_class[i]:.4f}")
                print(f"  Precision: {clf_report[label]['precision']:.4f}")
                print(f"  Recall: {clf_report[label]['recall']:.4f}")
                print(f"  Avg Precision: {avg_precision_scores[label]:.4f}")
            
            print(f"\n{'='*50}\n")
        
        # SALVAR PLOTS E RESULTADOS        
        if output_dir and save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Salvar JSON
            results_file = output_path / "detailed_results.json"
            results_to_save = {k: v for k, v in results.items() 
                            if k not in ['predictions', 'true_labels']}
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_to_save, f, indent=2, ensure_ascii=False)
            
            if verbose:
                print(f"✓ Resultados salvos em {results_file}")
        
        return results
    
def compute_ece(y_true, y_probs, n_bins=10):
    """
    Calcula Expected Calibration Error
    
    Args:
        y_true: Labels verdadeiros
        y_probs: Probabilidades preditas
        n_bins: Número de bins
        
    Returns:
        ece: Expected Calibration Error
    """
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece
