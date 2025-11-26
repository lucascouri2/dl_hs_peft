"""
Sistema de Fine tuning de modelos de linguagem
para detecção de Discurso de Ódio em Português Brasileiro
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    pipeline
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    PromptTuningConfig,
    PromptTuningInit
)
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ==================== CONFIGURAÇÕES ====================

@dataclass
class ExperimentConfig:
    """
    Configurações gerais do experimento
    """
    # Paths - OBRIGATÓRIOS
    data_path: str
    output_path: str
    cache_dir: str
    
    # Configurações gerais - OBRIGATÓRIAS
    seed: int
    max_length: int
    
    # Training - OBRIGATÓRIAS
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    early_stopping_patience: int
    
    # Quantização - OBRIGATÓRIAS
    use_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str
    use_nested_quant: bool
    
    # LoRA - OBRIGATÓRIAS
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: str
    lora_target_modules: List[str]
    
    # Prompt Tuning - OBRIGATÓRIAS
    num_virtual_tokens: int
    prompt_tuning_init: str
    
    # Otimizações - OBRIGATÓRIAS
    gradient_checkpointing: bool
    optim: str
    max_grad_norm: float
    
    # COM PADRÃO (sempre iguais)
    num_labels: int = 2
    label_names: List[str] = None
    
    def __post_init__(self):
        if self.label_names is None:
            self.label_names = ["NON-HATE", "HATE"]
        
        # Criar diretórios
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Set seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


# ==================== PRÉ-PROCESSAMENTO ====================

class TextPreprocessor:
    """Pré-processamento de textos"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'http\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        text = self.url_pattern.sub('<url>', text)
        text = self.mention_pattern.sub('<user>', text)
        text = self.email_pattern.sub('<email>', text)
        text = self.hashtag_pattern.sub(r'\1', text)
        text = text.replace('&amp;', ' and ').replace('&', ' and ')
        text = self.whitespace_pattern.sub(' ', text)
        text = text.strip()
        
        return text
    
    def preprocess_dataset(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        df = df.copy()
        df[text_column] = df[text_column].apply(self.clean_text)
        return df


# ==================== GERENCIAMENTO DE DADOS ====================

class DataManager:
    """Gerencia carregamento e preparação dos dados"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.preprocessor = TextPreprocessor()
        
    def load_data(self, train_path: str, val_path: str, test_path: str,
                  text_column: str = 'text', label_column: str = 'label') -> DatasetDict:
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Dados carregados:")
        print(f"  Train: {len(train_df)} exemplos")
        print(f"  Val: {len(val_df)} exemplos")
        print(f"  Test: {len(test_df)} exemplos")
        
        train_df = train_df.drop_duplicates(subset=[text_column])
        val_df = val_df.drop_duplicates(subset=[text_column])
        
        print(f"\nApós remoção de duplicatas:")
        print(f"  Train: {len(train_df)} exemplos")
        print(f"  Val: {len(val_df)} exemplos")
        
        train_df = self.preprocessor.preprocess_dataset(train_df, text_column)
        val_df = self.preprocessor.preprocess_dataset(val_df, text_column)
        test_df = self.preprocessor.preprocess_dataset(test_df, text_column)
        
        self._analyze_class_distribution(train_df, label_column)
        
        train_dataset = Dataset.from_pandas(train_df[[text_column, label_column]])
        val_dataset = Dataset.from_pandas(val_df[[text_column, label_column]])
        test_dataset = Dataset.from_pandas(test_df[[text_column, label_column]])
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
    
    def _analyze_class_distribution(self, df: pd.DataFrame, label_column: str):
        print("\n" + "="*50)
        print("DISTRIBUIÇÃO DE CLASSES (Training Set)")
        print("="*50)
        
        class_counts = df[label_column].value_counts().sort_index()
        total = len(df)
        
        for label, count in class_counts.items():
            label_name = self.config.label_names[label]
            percentage = (count / total) * 100
            print(f"{label_name} ({label}): {count} ({percentage:.2f}%)")
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(df[label_column]),
            y=df[label_column]
        )
        
        print("\nPesos calculados para balanceamento:")
        for label, weight in enumerate(class_weights):
            label_name = self.config.label_names[label]
            print(f"{label_name}: {weight:.4f}")
        
        print("="*50 + "\n")
        
        return class_weights
    
    def compute_class_weights(self, dataset: Dataset, label_column: str = 'label') -> torch.Tensor:
        labels = dataset[label_column]
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.tensor(class_weights, dtype=torch.float32)


# ==================== MODELOS ====================

class ModelFactory:
    """Factory para criar diferentes tipos de modelos"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def create_slm_model(self, model_name: str):
        print(f"\n[SLM] Carregando modelo: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.config.cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.config.num_labels,
            cache_dir=self.config.cache_dir
        )
        
        return model, tokenizer
    
    def create_llm_with_lora(self, model_name: str):
        print(f"\n[LLM-LoRA] Carregando modelo: {model_name}")
        
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.config.cache_dir)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.config.num_labels,
            quantization_config=bnb_config,
            cache_dir=self.config.cache_dir,
            device_map="auto",
            torch_dtype=torch.float16 if self.config.use_4bit else torch.float32
        )
        
        model.config.pad_token_id = tokenizer.pad_token_id
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            target_modules=self.config.lora_target_modules
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model, tokenizer
    
    def create_llm_with_prompt_tuning(self, model_name: str, prompt_text: str = None):
        print(f"\n[LLM-Prompt Tuning] Carregando modelo: {model_name}")
        
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.config.cache_dir)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.config.num_labels,
            quantization_config=bnb_config,
            cache_dir=self.config.cache_dir,
            device_map="auto",
            #torch_dtype=torch.float16 if self.config.use_4bit else torch.float32
            torch_dtype=torch.bfloat16 if self.config.use_4bit else torch.float32
        )
        
        model.config.pad_token_id = tokenizer.pad_token_id
        
        if prompt_text is None:
            #prompt_text = "Classifique o texto como discurso de ódio (atribuindo '1') ou não (atribuindo '0'):"
            #prompt_text = "Classify whether the following sentence contains hate speech. Answer with a single token '0' (no) or '1' (yes).\nText:"
            prompt_text = """Classifique se o texto contém discurso de ódio.
                            Responda apenas com o número:
                            - 0: NÃO contém discurso de ódio
                            - 1: CONTÉM discurso de ódio

                            Discurso de ódio inclui: xingamentos baseados em raça, religião, gênero, orientação sexual, nacionalidade ou outras características protegidas; ameaças; incitação à violência; desumanização.

                            Discurso de ódio NÃO inclui: críticas, opiniões contrárias, palavrões genéricos sem alvo específico, sarcasmo.

                            Texto:"""

        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=self.config.num_virtual_tokens,
            prompt_tuning_init=PromptTuningInit.TEXT if self.config.prompt_tuning_init == "TEXT" else PromptTuningInit.RANDOM,
            prompt_tuning_init_text=prompt_text if self.config.prompt_tuning_init == "TEXT" else None,
            tokenizer_name_or_path=model_name,
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model, tokenizer


# ==================== TREINAMENTO ====================

class HateSpeechTrainer:
    """Gerencia o treinamento dos modelos"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model_factory = ModelFactory(config)
        
    def tokenize_dataset(self, dataset: DatasetDict, tokenizer, 
                        text_column: str = 'text', label_column: str = 'label'):
        
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples[text_column],
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length
            )
            tokenized['labels'] = examples[label_column]
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset['train'].column_names if col != label_column]
        )
        
        return tokenized_dataset
    
    def train_slm(self, model_name: str, dataset: DatasetDict, 
                  output_dir: str, class_weights: torch.Tensor = None):
        
        model, tokenizer = self.model_factory.create_slm_model(model_name)
        tokenized_dataset = self.tokenize_dataset(dataset, tokenizer)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        if class_weights is not None:
            class_weights = class_weights.to(device)
        
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                
                if class_weights is not None:
                    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                else:
                    loss_fct = torch.nn.CrossEntropyLoss()
                
                loss = loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            seed=self.config.seed,
            fp16=torch.cuda.is_available(),
            report_to="none",
            push_to_hub=False,
            hub_strategy="end",
            hub_model_id=None,
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )
        
        print(f"\n{'='*50}")
        print(f"Iniciando treinamento SLM: {model_name}")
        print(f"{'='*50}\n")
        
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return trainer, model, tokenizer
    
    def train_llm_lora(self, model_name: str, dataset: DatasetDict,
                       output_dir: str, class_weights: torch.Tensor = None):
        
        model, tokenizer = self.model_factory.create_llm_with_lora(model_name)
        tokenized_dataset = self.tokenize_dataset(dataset, tokenizer)
        
        if class_weights is not None:
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            class_weights = class_weights.to(device=device, dtype=dtype)
        
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                
                if class_weights is not None:
                    #loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                    loss_fct = torch.nn.CrossEntropyLoss(
                        weight=class_weights.to(logits.device).to(logits.dtype)
                    )
                else:
                    loss_fct = torch.nn.CrossEntropyLoss()
                
                loss = loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            seed=self.config.seed,
            fp16=False,
            bf16=False,
            report_to="none",
            push_to_hub=False,
            hub_strategy="end",
            hub_model_id=None,
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )
        
        print(f"\n{'='*50}")
        print(f"Iniciando treinamento LLM-LoRA: {model_name}")
        print(f"{'='*50}\n")
        
        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return trainer, model, tokenizer
    
    def train_llm_prompt_tuning(self, model_name: str, dataset: DatasetDict,
                                output_dir: str, prompt_text: str = None,
                                class_weights: torch.Tensor = None):
        
        model, tokenizer = self.model_factory.create_llm_with_prompt_tuning(model_name, prompt_text)
        tokenized_dataset = self.tokenize_dataset(dataset, tokenizer)
        
        if class_weights is not None:
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            class_weights = class_weights.to(device=device, dtype=dtype)
        
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                
                if class_weights is not None:
                    #loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                    loss_fct = torch.nn.CrossEntropyLoss(
                        weight=class_weights.to(logits.device).to(logits.dtype)
                    )
                else:
                    loss_fct = torch.nn.CrossEntropyLoss()
                
                loss = loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            seed=self.config.seed,
            fp16=False,
            bf16=False,
            report_to="none",
            push_to_hub=False,
            hub_strategy="end",
            hub_model_id=None,
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )
        
        print(f"\n{'='*50}")
        print(f"Iniciando treinamento LLM-Prompt Tuning: {model_name}")
        print(f"{'='*50}\n")
        
        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return trainer, model, tokenizer
    
    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_per_class = f1_score(labels, predictions, average=None)
        accuracy = accuracy_score(labels, predictions)
        
        metrics = {
            'f1': f1_macro,
            'accuracy': accuracy,
        }
        
        for i, f1 in enumerate(f1_per_class):
            metrics[f'f1_class_{i}'] = f1
        
        return metrics


# ==================== AVALIAÇÃO ====================

class ModelEvaluator:
    """
    Avalia modelos com métricas robustas para datasets desbalanceados
    
    Métricas incluídas:
    - Macro-F1
    - F1 por classe
    - Precision-Recall Curve
    - Expected Calibration Error - ECE
    - Average Precision Score
    - ROC-AUC
    """
    
    def __init__(self, config):
        self.config = config
    
    def _compute_ece(self, y_true, y_probs, n_bins=10):
        """
        Calcula Expected Calibration Error (ECE)
        
        ECE mede o quão bem calibradas estão as probabilidades do modelo.
        Valores menores = melhor calibração
        
        Args:
            y_true: Labels verdadeiros
            y_probs: Probabilidades preditas (softmax)
            n_bins: Número de bins para calibração
            
        Returns:
            ece: Expected Calibration Error
            bin_data: Dados dos bins (para plotar)
        """
        # Pegar probabilidade da classe predita
        y_pred = np.argmax(y_probs, axis=1)
        confidences = np.max(y_probs, axis=1)
        accuracies = (y_pred == y_true).astype(float)
        
        # Criar bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Encontrar exemplos neste bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                bin_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'confidence': avg_confidence_in_bin,
                    'accuracy': accuracy_in_bin,
                    'count': in_bin.sum()
                })
            else:
                bin_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'confidence': 0,
                    'accuracy': 0,
                    'count': 0
                })
        
        return ece, bin_data
    
    def _plot_calibration_curve(self, bin_data, output_path, model_name):
        """
        Plota curva de calibração (reliability diagram)
        
        Args:
            bin_data: Dados dos bins do ECE
            output_path: Caminho para salvar figura
            model_name: Nome do modelo
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Extrair dados
        confidences = [b['confidence'] for b in bin_data if b['count'] > 0]
        accuracies = [b['accuracy'] for b in bin_data if b['count'] > 0]
        counts = [b['count'] for b in bin_data if b['count'] > 0]
        
        # Plot
        ax.bar(range(len(confidences)), accuracies, 
               width=0.1, alpha=0.7, label='Accuracy', edgecolor='black')
        ax.plot(range(len(confidences)), confidences, 
                'r--', marker='o', label='Confidence', linewidth=2)
        
        # Linha de calibração perfeita
        ax.plot([0, len(confidences)-1], [0, 1], 'k--', alpha=0.3, label='Perfect Calibration')
        
        ax.set_xlabel('Bin', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_title(f'Calibration Curve - {model_name}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, y_true, y_probs, output_path, model_name):
        """
        Plota Precision-Recall Curve para cada classe
        
        Args:
            y_true: Labels verdadeiros
            y_probs: Probabilidades preditas
            output_path: Caminho para salvar
            model_name: Nome do modelo
        """
        n_classes = y_probs.shape[1]
        
        fig, axes = plt.subplots(1, n_classes, figsize=(6*n_classes, 5))
        if n_classes == 1:
            axes = [axes]
        
        for i, class_name in enumerate(self.config.label_names):
            # Binarizar labels para esta classe
            y_true_binary = (y_true == i).astype(int)
            y_scores = y_probs[:, i]
            
            # Calcular precision-recall
            precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
            avg_precision = average_precision_score(y_true_binary, y_scores)
            
            # Plot
            axes[i].plot(recall, precision, linewidth=2, 
                        label=f'AP = {avg_precision:.3f}')
            axes[i].fill_between(recall, precision, alpha=0.2)
            
            axes[i].set_xlabel('Recall', fontsize=12)
            axes[i].set_ylabel('Precision', fontsize=12)
            axes[i].set_title(f'{class_name}\nPrecision-Recall Curve', fontsize=12)
            axes[i].legend(loc='best')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim([0, 1])
            axes[i].set_ylim([0, 1])
        
        plt.suptitle(f'Precision-Recall Curves - {model_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, cm, output_path, model_name):
        """
        Plota matriz de confusão normalizada e absoluta
        
        Args:
            cm: Confusion matrix
            output_path: Caminho para salvar
            model_name: Nome do modelo
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Normalizada (por linha - recall)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot absoluta
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.config.label_names,
                   yticklabels=self.config.label_names,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Absolute)', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=11)
        axes[0].set_xlabel('Predicted Label', fontsize=11)
        
        # Plot normalizada
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.config.label_names,
                   yticklabels=self.config.label_names,
                   ax=axes[1], cbar_kws={'label': 'Proportion'})
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=11)
        axes[1].set_xlabel('Predicted Label', fontsize=11)
        
        plt.suptitle(f'Confusion Matrices - {model_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_model(self, model, tokenizer, test_dataset,
                      output_dir: str, model_name: str, 
                      save_plots: bool = True):
        """
        Avalia modelo com métricas completas
        
        Args:
            model: Modelo treinado
            tokenizer: Tokenizer
            test_dataset: Dataset de teste
            output_dir: Diretório para salvar resultados
            model_name: Nome do modelo
            save_plots: Se True, salva gráficos
            
        Returns:
            Dicionário com todas as métricas
        """
        print(f"\n{'='*50}")
        print(f"Avaliando modelo: {model_name}")
        print(f"{'='*50}\n")
        
        texts = test_dataset['text']
        true_labels = test_dataset['label']
        
        predictions = []
        probabilities = []
        
        model.eval()
        
        print("Gerando predições...")
        with torch.no_grad():
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                ).to(model.device)
                
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Probabilidades (softmax)
                #probs = torch.softmax(logits, dim=-1).cpu().numpy()
                # Probabilidades (softmax) - CONVERTER PARA FLOAT32
                probs = torch.softmax(logits, dim=-1).float().cpu().numpy()  # mudei pra esse após usar bfloat16
                probabilities.extend(probs)
                
                # Predições
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                predictions.extend(preds)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        true_labels = np.array(true_labels)
        
        print("Calculando métricas...")
        
        # ===================================================================
        # MÉTRICAS BÁSICAS
        # ===================================================================
        
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        f1_per_class = f1_score(true_labels, predictions, average=None)
        accuracy = accuracy_score(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions)
        
        # ===================================================================
        # MÉTRICAS PARA DATASETS DESBALANCEADOS
        # ===================================================================
        
        # Average Precision Score (melhor que accuracy para desbalanceado)
        avg_precision_scores = {}
        for i, label_name in enumerate(self.config.label_names):
            y_true_binary = (true_labels == i).astype(int)
            y_scores = probabilities[:, i]
            avg_precision_scores[label_name] = average_precision_score(y_true_binary, y_scores)
        
        # Macro Average Precision
        macro_avg_precision = np.mean(list(avg_precision_scores.values()))
        
        # ROC-AUC (se binário)
        if len(self.config.label_names) == 2:
            roc_auc = roc_auc_score(true_labels, probabilities[:, 1])
        else:
            # Multi-class: one-vs-rest
            roc_auc = roc_auc_score(true_labels, probabilities, 
                                    multi_class='ovr', average='macro')
        
        # ===================================================================
        # EXPECTED CALIBRATION ERROR (ECE)
        # ===================================================================
        
        ece, bin_data = self._compute_ece(true_labels, probabilities, n_bins=10)
        
        # ===================================================================
        # PRECISION, RECALL, F1 POR CLASSE (detalhado)
        # ===================================================================
        
        clf_report = classification_report(
            true_labels, predictions,
            target_names=self.config.label_names,
            output_dict=True,
            zero_division=0
        )
        
        # ===================================================================
        # MONTAR RESULTADOS
        # ===================================================================
        
        results = {
            # Métricas principais
            'macro_f1': float(macro_f1),
            'accuracy': float(accuracy),
            'f1_per_class': {
                self.config.label_names[i]: float(f1) 
                for i, f1 in enumerate(f1_per_class)
            },
            
            # Métricas para desbalanceamento
            'macro_avg_precision': float(macro_avg_precision),
            'avg_precision_per_class': {
                k: float(v) for k, v in avg_precision_scores.items()
            },
            'roc_auc': float(roc_auc),
            
            # Calibração
            'expected_calibration_error': float(ece),
            
            # Detalhes por classe
            'precision_per_class': {
                label: float(clf_report[label]['precision'])
                for label in self.config.label_names
            },
            'recall_per_class': {
                label: float(clf_report[label]['recall'])
                for label in self.config.label_names
            },
            
            # Confusion matrix
            'confusion_matrix': cm.tolist(),
            
            # Report completo
            'classification_report': clf_report,
            
            # Dados brutos (para análise posterior)
            'predictions': predictions.tolist(),
            'true_labels': true_labels.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        # ===================================================================
        # IMPRIMIR RESULTADOS
        # ===================================================================
        
        print(f"\n{'='*50}")
        print("RESULTADOS")
        print(f"{'='*50}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro Avg Precision: {macro_avg_precision:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Expected Calibration Error: {ece:.4f}")
        
        print(f"\n{'='*50}")
        print("MÉTRICAS POR CLASSE")
        print(f"{'='*50}")
        
        for i, label in enumerate(self.config.label_names):
            print(f"\n{label}:")
            print(f"  F1: {f1_per_class[i]:.4f}")
            print(f"  Precision: {clf_report[label]['precision']:.4f}")
            print(f"  Recall: {clf_report[label]['recall']:.4f}")
            print(f"  Avg Precision (AP): {avg_precision_scores[label]:.4f}")
        
        print(f"\n{'='*50}\n")
        
        # ===================================================================
        # SALVAR GRÁFICOS
        # ===================================================================
        
        if save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            print("Gerando visualizações...")
            
            # Confusion Matrix
            self._plot_confusion_matrix(
                cm, 
                output_path / f"{model_name}_confusion_matrix.png",
                model_name
            )
            
            # Precision-Recall Curves
            self._plot_precision_recall_curve(
                true_labels,
                probabilities,
                output_path / f"{model_name}_precision_recall.png",
                model_name
            )
            
            # Calibration Curve
            self._plot_calibration_curve(
                bin_data,
                output_path / f"{model_name}_calibration.png",
                model_name
            )
            
            print(f"✓ Visualizações salvas em {output_path}")
        
        # ===================================================================
        # SALVAR RESULTADOS JSON
        # ===================================================================
        
        output_path = Path(output_dir)
        results_file = output_path / f"{model_name}_detailed_results.json"
        
        # Versão serializable (sem numpy arrays grandes)
        results_to_save = {k: v for k, v in results.items() 
                          if k not in ['predictions', 'true_labels', 'probabilities']}
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Resultados detalhados salvos em {results_file}")
        
        return results


# ============================================================================
# FUNÇÃO AUXILIAR: COMPARAR MÚLTIPLOS MODELOS
# ============================================================================

def compare_models_advanced(results_dict: dict, output_path: str = "./comparison"):
    """
    Compara múltiplos modelos com métricas avançadas
    
    Args:
        results_dict: Dict com {model_name: results}
        output_path: Onde salvar comparação
    """
    import pandas as pd
    
    comparison_data = []
    
    for model_name, results in results_dict.items():
        row = {
            'model': model_name,
            'macro_f1': results['macro_f1'],
            'accuracy': results['accuracy'],
            'macro_avg_precision': results['macro_avg_precision'],
            'roc_auc': results['roc_auc'],
            'ece': results['expected_calibration_error'],
        }
        
        # Adicionar F1 por classe
        for class_name, f1 in results['f1_per_class'].items():
            row[f'f1_{class_name}'] = f1
        
        # Adicionar Avg Precision por classe
        for class_name, ap in results['avg_precision_per_class'].items():
            row[f'ap_{class_name}'] = ap
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Ordenar por Macro-F1
    df = df.sort_values('macro_f1', ascending=False)
    
    # Salvar
    Path(output_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{output_path}/advanced_comparison.csv", index=False)
    
    # Imprimir
    print(f"\n{'='*70}")
    print("COMPARAÇÃO AVANÇADA DE MODELOS")
    print(f"{'='*70}\n")
    
    # Métricas principais
    print("Métricas Principais:")
    print(df[['model', 'macro_f1', 'macro_avg_precision', 'roc_auc', 'ece']].to_string(index=False))
    
    print(f"\n{'='*70}")
    print(f"✓ Comparação salva em: {output_path}/advanced_comparison.csv")
    print(f"{'='*70}\n")
    
    return df

# ==================== EXEMPLO DE USO ====================

if __name__ == "__main__":
    
    # AGORA COM ARGUMENTOS EXPLÍCITOS
    config = ExperimentConfig(
        data_path="./data",
        output_path="./results",
        cache_dir="./cache",
        seed=42,
        max_length=256,
        batch_size=16,
        num_epochs=3,
        learning_rate=1e-4,
        weight_decay=0.0001,
        warmup_ratio=0.1,
        gradient_accumulation_steps=2,
        early_stopping_patience=5,
        use_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        use_nested_quant=True,
        lora_r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_bias="none",
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        num_virtual_tokens=20,
        prompt_tuning_init="TEXT",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
    )
    
    data_manager = DataManager(config)
    trainer = HateSpeechTrainer(config)
    evaluator = ModelEvaluator(config)
    
    dataset = data_manager.load_data(
        train_path="./data/train.csv",
        val_path="./data/val.csv",
        test_path="./data/test.csv"
    )
    
    class_weights = data_manager.compute_class_weights(dataset['train'])
    
    # Treinar
    trainer_obj, model, tokenizer = trainer.train_slm(
        model_name='neuralmind/bert-base-portuguese-cased',
        dataset=dataset,
        output_dir='./results/bertimbau',
        class_weights=class_weights
    )
    
    # Avaliar
    results = evaluator.evaluate_model(
        model, tokenizer, dataset['test'],
        output_dir='./results/bertimbau',
        model_name='bertimbau'
    )