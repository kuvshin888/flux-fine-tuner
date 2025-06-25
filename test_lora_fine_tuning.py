#!/usr/bin/env python3
"""
Простой тест для проверки функциональности дообучения LoRA
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Добавляем текущую директорию в путь для импорта train.py
import sys
sys.path.insert(0, '.')

from train import download_pretrained_lora_for_training, INPUT_DIR


class TestLoRAFineTuning(unittest.TestCase):
    
    def setUp(self):
        """Подготовка для тестов"""
        self.test_url = "https://huggingface.co/test/model/resolve/main/lora.safetensors"
        self.invalid_url = "https://example.com/invalid.txt"
    
    def test_valid_lora_url(self):
        """Тест валидного URL для LoRA"""
        # Мокаем subprocess.check_output и существование файла
        with patch('train.subprocess.check_output') as mock_subprocess, \
             patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'stat') as mock_stat:
            
            mock_stat.return_value.st_size = 1024 * 1024  # 1MB
            
            result = download_pretrained_lora_for_training(self.test_url)
            
            # Проверяем, что функция возвращает Path
            self.assertIsInstance(result, Path)
            self.assertTrue(str(result).endswith('pretrained_lora.safetensors'))
            
            # Проверяем, что subprocess был вызван с правильными аргументами
            mock_subprocess.assert_called_once()
            args = mock_subprocess.call_args[0][0]
            self.assertIn('pget', args)
            self.assertIn(self.test_url, args)
    
    def test_invalid_url_format(self):
        """Тест невалидного URL"""
        with self.assertRaises(ValueError) as context:
            download_pretrained_lora_for_training(self.invalid_url)
        
        self.assertIn("Invalid URL", str(context.exception))
        self.assertIn("HuggingFace download URL", str(context.exception))
    
    def test_invalid_file_extension(self):
        """Тест URL с неправильным расширением файла"""
        invalid_ext_url = "https://huggingface.co/test/model/resolve/main/lora.bin"
        
        with self.assertRaises(ValueError) as context:
            download_pretrained_lora_for_training(invalid_ext_url)
        
        self.assertIn("Invalid URL", str(context.exception))
    
    def test_download_failure(self):
        """Тест неудачной загрузки"""
        with patch('train.subprocess.check_output') as mock_subprocess, \
             patch.object(Path, 'exists', return_value=False):
            
            with self.assertRaises(RuntimeError) as context:
                download_pretrained_lora_for_training(self.test_url)
            
            self.assertIn("Failed to download", str(context.exception))


def test_train_parameters():
    """Простой тест для проверки добавленных параметров в функции train"""
    from train import train
    import inspect
    
    # Получаем сигнатуру функции train
    sig = inspect.signature(train)
    
    # Проверяем, что новые параметры добавлены
    params = list(sig.parameters.keys())
    
    assert 'pretrained_lora_url' in params, "Параметр pretrained_lora_url не найден"
    assert 'keep_optimizer_for_resume' in params, "Параметр keep_optimizer_for_resume не найден"
    
    # Проверяем значения по умолчанию
    assert sig.parameters['pretrained_lora_url'].default is None
    assert sig.parameters['keep_optimizer_for_resume'].default is False
    
    print("✅ Все параметры функции train корректны!")


if __name__ == "__main__":
    print("🧪 Запуск тестов функциональности дообучения LoRA...")
    
    # Запускаем простой тест параметров
    test_train_parameters()
    
    # Запускаем unittest'ы
    unittest.main(verbosity=2) 