# Дообучение готовых LoRA в FLUX Fine-Tuner

Теперь вы можете дообучать уже готовые LoRA вместо создания новых с нуля! Это полезно для:

- 🎨 **Улучшения недообученных LoRA** - добавьте больше шагов обучения
- 🔄 **Адаптации под новые стили** - возьмите портретную LoRA и дообучите на пейзажах  
- 🎯 **Комбинирования концептов** - смешайте несколько стилей в одной LoRA
- ⚡ **Экономии времени** - не начинайте обучение с нуля

## Новые параметры

### `pretrained_lora_url`
URL готовой LoRA для дообучения. Поддерживает HuggingFace URLs в формате:
```
https://huggingface.co/user/model/resolve/main/lora.safetensors
```

**Пример:**
```
https://huggingface.co/fofr/flux-80s-cyberpunk/resolve/main/lora.safetensors
```

### `keep_optimizer_for_resume` 
Сохранить `optimizer.pt` файл после обучения для возможности будущего дообучения.
- `true` - сохранить optimizer для возможности продолжения обучения
- `false` (по умолчанию) - удалить optimizer для экономии места

## Примеры использования

### Базовое дообучение
```bash
# Дообучить готовую LoRA на новых данных
python train.py \
  --input_images="my_new_images.zip" \
  --pretrained_lora_url="https://huggingface.co/user/model/resolve/main/lora.safetensors" \
  --trigger_word="NEWSTYLE" \
  --steps=500 \
  --learning_rate=2e-4
```

### Улучшение недообученной LoRA
```bash
# Продолжить обучение недообученной LoRA
python train.py \
  --input_images="original_dataset.zip" \
  --pretrained_lora_url="https://huggingface.co/myuser/underfitted-lora/resolve/main/lora.safetensors" \
  --trigger_word="SAMESTYLE" \
  --steps=1000 \
  --learning_rate=1e-4 \
  --keep_optimizer_for_resume=true
```

### Адаптация стиля
```bash
# Взять портретную LoRA и адаптировать под пейзажи
python train.py \
  --input_images="landscape_photos.zip" \
  --pretrained_lora_url="https://huggingface.co/portraits/amazing-portraits/resolve/main/lora.safetensors" \
  --trigger_word="LANDSCP" \
  --steps=800 \
  --learning_rate=3e-4
```

## Логи и мониторинг

При дообучении LoRA вы увидите информативные логи:

```
🔄 Режим дообучения готовой LoRA активирован
Downloading pretrained LoRA from: https://huggingface.co/user/model/resolve/main/lora.safetensors
✅ Готовая LoRA загружена: /path/to/pretrained_lora.safetensors
🔗 Настраиваем дообучение с готовой LoRA: /path/to/pretrained_lora.safetensors
✅ Конфигурация для дообучения LoRA настроена

🚀 Запуск дообучения готовой LoRA
   📁 Исходная LoRA: /path/to/pretrained_lora.safetensors
   🎯 Новый триггер: NEWSTYLE
   📊 Шагов обучения: 500
   📈 Learning rate: 2e-4

...

🎉 Дообучение LoRA завершено успешно!
   📁 Исходная LoRA: https://huggingface.co/user/model/resolve/main/lora.safetensors
   ✨ Результирующая LoRA: 15.2 MB
   🎯 Триггер-слово: NEWSTYLE
   📊 Выполнено шагов: 500
```

## Weights & Biases интеграция

При использовании W&B, дополнительная информация о дообучении будет логироваться:
- `pretrained_lora_url` - URL исходной LoRA
- `is_fine_tuning_mode` - флаг режима дообучения
- `keep_optimizer_for_resume` - сохранен ли optimizer

## Важные замечания

1. **Формат файлов**: Поддерживаются только `.safetensors` файлы
2. **Источники**: Только HuggingFace URLs в данный момент
3. **Конфликты**: Нельзя одновременно использовать `pretrained_lora_url` и `skip_training_and_use_pretrained_hf_lora_url`
4. **Параметры**: Можно изменять любые параметры обучения (learning rate, steps, trigger word и т.д.)
5. **Память**: Дообучение требует столько же памяти, сколько обычное обучение

## Технические детали

- Готовая LoRA загружается перед началом обучения
- Веса LoRA используются для инициализации новой сети
- Это **новое обучение**, а не resume прерванного процесса
- Optimizer и scheduler начинают с нуля
- Все параметры обучения могут быть изменены

Удачного дообучения! 🚀 