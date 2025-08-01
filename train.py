import os

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import sys

sys.path.append("ai-toolkit")
sys.path.append("LLaVA")

from submodule_patches import patch_submodules

patch_submodules()

import shutil
import subprocess
import sys
import time
from typing import Optional, OrderedDict
from zipfile import ZipFile, is_zipfile
from string import Template

import torch
from cog import BaseModel, Input, Path, Secret  # pyright: ignore

# ai-toolkit теперь установлен как пакет в editable режиме
try:
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    print("✅ Успешно импортирован SDTrainer из ai-toolkit")
except ImportError as e:
    print(f"❌ Ошибка импорта SDTrainer: {e}")
    # Fallback: попробуем добавить пути вручную
    import sys
    import os
    
    ai_toolkit_paths = [
        os.path.join(os.path.dirname(__file__), "ai-toolkit"),
        "/src/ai-toolkit"
    ]
    
    for path in ai_toolkit_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.append(path)
            print(f"📂 Добавлен путь: {path}")
    
    print(f"🔍 Текущие пути Python: {[p for p in sys.path if 'ai-toolkit' in p]}")
    
    # Попробуем еще раз
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    print("✅ Импорт SDTrainer успешен после добавления путей")
from huggingface_hub import HfApi
from jobs import BaseJob
from toolkit.config import get_config

from caption import Captioner
from wandb_client import WeightsAndBiasesClient, logout_wandb
from layer_match import match_layers_to_optimize, available_layers_to_optimize


JOB_NAME = "flux_train_replicate"
WEIGHTS_PATH = Path("./FLUX.1-dev")
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("output")
JOB_DIR = OUTPUT_DIR / JOB_NAME

print(f"Environment language: {os.environ.get('LANG', 'Not set')}")
os.environ["LANG"] = "en_US.UTF-8"
print(f"Updated environment language: {os.environ.get('LANG', 'Not set')}")


class CustomSDTrainer(SDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_samples = set()
        self.wandb: WeightsAndBiasesClient | None = None

    def hook_train_loop(self, batch):
        loss_dict = super().hook_train_loop(batch)
        if self.wandb:
            self.wandb.log_loss(loss_dict, self.step_num)
        return loss_dict

    def sample(self, step=None, is_first=False):
        super().sample(step=step, is_first=is_first)
        output_dir = JOB_DIR / "samples"
        all_samples = set([p.name for p in output_dir.glob("*.jpg")])
        new_samples = all_samples - self.seen_samples
        if self.wandb:
            image_paths = [output_dir / p for p in sorted(new_samples)]
            self.wandb.log_samples(image_paths, step)
        self.seen_samples = all_samples

    def post_save_hook(self, save_path):
        super().post_save_hook(save_path)
        # final lora path
        lora_path = JOB_DIR / f"{JOB_NAME}.safetensors"
        if not lora_path.exists():
            # intermediate saved weights
            lora_path = sorted(JOB_DIR.glob("*.safetensors"))[-1]
        if self.wandb:
            print(f"Saving weights to W&B: {lora_path.name}")
            self.wandb.save_weights(lora_path)


class CustomJob(BaseJob):
    def __init__(
        self, config: OrderedDict, wandb_client: WeightsAndBiasesClient | None
    ):
        super().__init__(config)
        self.device = self.get_conf("device", "cpu")
        self.process_dict = {"custom_sd_trainer": CustomSDTrainer}
        self.load_processes(self.process_dict)
        for process in self.process:
            process.wandb = wandb_client

    def run(self):
        super().run()
        # Keeping this for backwards compatibility
        print(
            f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}"
        )
        for process in self.process:
            process.run()


class TrainingOutput(BaseModel):
    weights: Path


def train(
    input_images: Path = Input(
        description="A zip file containing the images that will be used for training. We recommend a minimum of 10 images. If you include captions, include them as one .txt file per image, e.g. my-photo.jpg should have a caption file named my-photo.txt. If you don't include captions, you can use autocaptioning (enabled by default).",
        default=None,
    ),
    trigger_word: str = Input(
        description="The trigger word refers to the object, style or concept you are training on. Pick a string that isn't a real word, like TOK or something related to what's being trained, like CYBRPNK. The trigger word you specify here will be associated with all images during training. Then when you use your LoRA, you can include the trigger word in prompts to help activate the LoRA.",
        default="TOK",
    ),
    autocaption: bool = Input(
        description="Automatically caption images using Llava v1.5 13B", default=True
    ),
    autocaption_prefix: str = Input(
        description="Optional: Text you want to appear at the beginning of all your generated captions; for example, 'a photo of TOK, '. You can include your trigger word in the prefix. Prefixes help set the right context for your captions, and the captioner will use this prefix as context.",
        default=None,
    ),
    autocaption_suffix: str = Input(
        description="Optional: Text you want to appear at the end of all your generated captions; for example, ' in the style of TOK'. You can include your trigger word in suffixes. Suffixes help set the right concept for your captions, and the captioner will use this suffix as context.",
        default=None,
    ),
    steps: int = Input(
        description="Number of training steps. Recommended range 500-4000",
        ge=3,
        le=6000,
        default=1000,
    ),
    learning_rate: float = Input(
        description="Learning rate, if you're new to training you probably don't need to change this.",
        default=4e-4,
    ),
    batch_size: int = Input(
        description="Batch size, you can leave this as 1", default=1
    ),
    resolution: str = Input(
        description="Image resolutions for training", default="512,768,1024"
    ),
    lora_rank: int = Input(
        description="Higher ranks take longer to train but can capture more complex features. Caption quality is more important for higher ranks.",
        default=16,
        ge=1,
        le=128,
    ),
    caption_dropout_rate: float = Input(
        description="Advanced setting. Determines how often a caption is ignored. 0.05 means for 5% of all steps an image will be used without its caption. 0 means always use captions, while 1 means never use them. Dropping captions helps capture more details of an image, and can prevent over-fitting words with specific image elements. Try higher values when training a style.",
        default=0.05,
        ge=0,
        le=1,
    ),
    optimizer: str = Input(
        description="Optimizer to use for training. Supports: prodigy, adam8bit, adamw8bit, lion8bit, adam, adamw, lion, adagrad, adafactor.",
        default="adamw8bit",
    ),
    cache_latents_to_disk: bool = Input(
        description="Use this if you have lots of input images and you hit out of memory errors",
        default=False,
    ),
    layers_to_optimize_regex: str = Input(
        description="Regular expression to match specific layers to optimize. Optimizing fewer layers results in shorter training times, but can also result in a weaker LoRA. For example, To target layers 7, 12, 16, 20 which seems to create good likeness with faster training (as discovered by lux in the Ostris discord, inspired by The Last Ben), use `transformer.single_transformer_blocks.(7|12|16|20).proj_out`.",
        default=None,
    ),
    gradient_checkpointing: bool = Input(
        description="Turn on gradient checkpointing; saves memory at the cost of training speed. Automatically enabled for batch sizes > 1.",
        default=False,
    ),
    hf_repo_id: str = Input(
        description="Hugging Face repository ID, if you'd like to upload the trained LoRA to Hugging Face. For example, lucataco/flux-dev-lora. If the given repo does not exist, a new public repo will be created.",
        default=None,
    ),
    hf_token: Secret = Input(
        description="Hugging Face token, if you'd like to upload the trained LoRA to Hugging Face.",
        default=None,
    ),
    wandb_api_key: Secret = Input(
        description="Weights and Biases API key, if you'd like to log training progress to W&B.",
        default=None,
    ),
    wandb_project: str = Input(
        description="Weights and Biases project name. Only applicable if wandb_api_key is set.",
        default=JOB_NAME,
    ),
    wandb_run: str = Input(
        description="Weights and Biases run name. Only applicable if wandb_api_key is set.",
        default=None,
    ),
    wandb_entity: str = Input(
        description="Weights and Biases entity name. Only applicable if wandb_api_key is set.",
        default=None,
    ),
    wandb_sample_interval: int = Input(
        description="Step interval for sampling output images that are logged to W&B. Only applicable if wandb_api_key is set.",
        default=100,
        ge=1,
    ),
    wandb_sample_prompts: str = Input(
        description="Newline-separated list of prompts to use when logging samples to W&B. Only applicable if wandb_api_key is set.",
        default=None,
    ),
    wandb_save_interval: int = Input(
        description="Step interval for saving intermediate LoRA weights to W&B. Only applicable if wandb_api_key is set.",
        default=100,
        ge=1,
    ),
    skip_training_and_use_pretrained_hf_lora_url: Optional[str] = Input(
        description="If you'd like to skip LoRA training altogether and instead create a Replicate model from a pre-trained LoRA that's on HuggingFace, use this field with a HuggingFace download URL. For example, https://huggingface.co/fofr/flux-80s-cyberpunk/resolve/main/lora.safetensors.",
        default=None,
    ),
    pretrained_lora_url: Optional[str] = Input(
        description="URL готовой LoRA для дообучения. Поддерживает HuggingFace URLs как https://huggingface.co/user/model/resolve/main/lora.safetensors. Если указано, будет продолжено обучение с этой LoRA вместо создания новой с нуля. Это полезно для улучшения недообученных LoRA или адаптации под новые стили.",
        default=None,
    ),
    keep_optimizer_for_resume: bool = Input(
        description="Сохранить optimizer.pt файл после обучения для возможности будущего дообучения. По умолчанию optimizer удаляется для экономии места.",
        default=False,
    ),
) -> TrainingOutput:
    clean_up()
    output_path = "/tmp/trained_model.tar"

    if skip_training_and_use_pretrained_hf_lora_url is not None:
        download_huggingface_lora(
            skip_training_and_use_pretrained_hf_lora_url, output_path
        )
        return TrainingOutput(weights=Path(output_path))
    if not input_images:
        raise ValueError("input_images must be provided")
    
    # Проверяем конфликт параметров
    if pretrained_lora_url and skip_training_and_use_pretrained_hf_lora_url:
        raise ValueError("Нельзя одновременно указывать pretrained_lora_url и skip_training_and_use_pretrained_hf_lora_url")
    
    # Загрузка готовой LoRA для дообучения
    pretrained_lora_path = None
    if pretrained_lora_url:
        print("🔄 Режим дообучения готовой LoRA активирован")
        pretrained_lora_path = download_pretrained_lora_for_training(pretrained_lora_url)
        print(f"✅ Готовая LoRA загружена: {pretrained_lora_path}")
    else:
        print("🆕 Режим обучения LoRA с нуля")

    layers_to_optimize = None
    if layers_to_optimize_regex:
        layers_to_optimize = match_layers_to_optimize(layers_to_optimize_regex)
        if not layers_to_optimize:
            raise ValueError(
                f"The regex '{layers_to_optimize_regex}' didn't match any layers. These layers can be optimized:\n"
                + "\n".join(available_layers_to_optimize)
            )
    quantize = False
    resolutions = [int(res) for res in resolution.split(",")]

    sample_prompts = []
    if wandb_sample_prompts:
        sample_prompts = [p.strip() for p in wandb_sample_prompts.split("\n")]

    if not gradient_checkpointing:
        if (
            torch.cuda.get_device_properties(0).total_memory
            < 1024 * 1024 * 1024 * 100  # memory < 100 GB?
        ):
            print(
                "Turning gradient checkpointing on and quantizing base model, GPU has less than 100 GB of memory"
            )
            gradient_checkpointing = True
            quantize = True
        elif batch_size > 1:
            print("Turning gradient checkpointing on automatically for batch size > 1")
            gradient_checkpointing = True
        elif max(resolutions) > 1024:
            print(
                "Turning gradient checkpointing on; training resolution greater than 1024x1024"
            )
            gradient_checkpointing = True

    train_config = OrderedDict(
        {
            "job": "custom_job",
            "config": {
                "name": JOB_NAME,
                "process": [
                    {
                        "type": "custom_sd_trainer",
                        "training_folder": str(OUTPUT_DIR),
                        "device": "cuda:0",
                        "trigger_word": trigger_word,
                        "network": {
                            "type": "lora",
                            "linear": lora_rank,
                            "linear_alpha": lora_rank,
                        },
                        "save": {
                            "dtype": "float16",
                            "save_every": (
                                wandb_save_interval if wandb_api_key else steps + 1
                            ),
                            "max_step_saves_to_keep": 1,
                        },
                        "datasets": [
                            {
                                "folder_path": str(INPUT_DIR),
                                "caption_ext": "txt",
                                "caption_dropout_rate": caption_dropout_rate,
                                "shuffle_tokens": False,
                                # TODO: Do we need to cache to disk? It's faster not to.
                                "cache_latents_to_disk": cache_latents_to_disk,
                                "cache_latents": True,
                                "resolution": resolutions,
                            }
                        ],
                        "train": {
                            "batch_size": batch_size,
                            "steps": steps,
                            "gradient_accumulation_steps": 1,
                            "train_unet": True,
                            "train_text_encoder": False,
                            "content_or_style": "balanced",
                            "gradient_checkpointing": gradient_checkpointing,
                            "noise_scheduler": "flowmatch",
                            "optimizer": optimizer,
                            "lr": learning_rate,
                            "ema_config": {"use_ema": True, "ema_decay": 0.99},
                            "dtype": "bf16",
                        },
                        "model": {
                            "name_or_path": str(WEIGHTS_PATH),
                            "is_flux": True,
                            "quantize": quantize,
                        },
                        "sample": {
                            "sampler": "flowmatch",
                            "sample_every": (
                                wandb_sample_interval
                                if wandb_api_key and sample_prompts
                                else steps + 1
                            ),
                            "width": 1024,
                            "height": 1024,
                            "prompts": sample_prompts,
                            "neg": "",
                            "seed": 42,
                            "walk_seed": True,
                            "guidance_scale": 3.5,
                            "sample_steps": 28,
                        },
                    }
                ],
            },
            "meta": {"name": "[name]", "version": "1.0"},
        }
    )

    # Добавляем поддержку предварительно обученной LoRA
    if pretrained_lora_path:
        print(f"🔗 Настраиваем дообучение с готовой LoRA: {pretrained_lora_path}")
        if "network_kwargs" not in train_config["config"]["process"][0]["network"]:
            train_config["config"]["process"][0]["network"]["network_kwargs"] = {}
        
        # Добавляем путь к готовой LoRA для инициализации весов
        train_config["config"]["process"][0]["network"]["network_kwargs"]["pretrained_path"] = str(pretrained_lora_path)
        
        print("✅ Конфигурация для дообучения LoRA настроена")

    if layers_to_optimize:
        if "network_kwargs" not in train_config["config"]["process"][0]["network"]:
            train_config["config"]["process"][0]["network"]["network_kwargs"] = {}
        train_config["config"]["process"][0]["network"]["network_kwargs"]["only_if_contains"] = layers_to_optimize

    wandb_client = None
    if wandb_api_key:
        wandb_config = {
            "trigger_word": trigger_word,
            "autocaption": autocaption,
            "autocaption_prefix": autocaption_prefix,
            "autocaption_suffix": autocaption_suffix,
            "steps": steps,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "resolution": resolution,
            "lora_rank": lora_rank,
            "caption_dropout_rate": caption_dropout_rate,
            "optimizer": optimizer,
            "pretrained_lora_url": pretrained_lora_url,
            "is_fine_tuning_mode": pretrained_lora_url is not None,
            "keep_optimizer_for_resume": keep_optimizer_for_resume,
        }
        wandb_client = WeightsAndBiasesClient(
            api_key=wandb_api_key.get_secret_value(),
            config=wandb_config,
            sample_prompts=sample_prompts,
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run,
        )

    download_weights()
    extract_zip(input_images, INPUT_DIR)

    if not trigger_word:
        del train_config["config"]["process"][0]["trigger_word"]

    captioner = Captioner()
    if autocaption and not captioner.all_images_are_captioned(INPUT_DIR):
        captioner.load_models()
        captioner.caption_images(INPUT_DIR, autocaption_prefix, autocaption_suffix)

    del captioner
    torch.cuda.empty_cache()

    # Информативный вывод о режиме обучения
    if pretrained_lora_path:
        print("🚀 Запуск дообучения готовой LoRA")
        print(f"   📁 Исходная LoRA: {pretrained_lora_path}")
        print(f"   🎯 Новый триггер: {trigger_word}")
        print(f"   📊 Шагов обучения: {steps}")
        print(f"   📈 Learning rate: {learning_rate}")
    else:
        print("🆕 Запуск обучения LoRA с нуля")
    
    print("Starting train job")
    job = CustomJob(get_config(train_config, name=None), wandb_client)
    job.run()

    if wandb_client:
        wandb_client.finish()

    job.cleanup()

    lora_file = JOB_DIR / f"{JOB_NAME}.safetensors"
    lora_file.rename(JOB_DIR / "lora.safetensors")

    samples_dir = JOB_DIR / "samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)

    # Remove any intermediate lora paths
    lora_paths = JOB_DIR.glob("*.safetensors")
    for path in lora_paths:
        if path.name != "lora.safetensors":
            path.unlink()

    # Управление optimizer.pt файлом
    optimizer_file = JOB_DIR / "optimizer.pt"
    if optimizer_file.exists():
        if keep_optimizer_for_resume:
            print(f"💾 Сохраняем optimizer.pt для возможности будущего дообучения: {optimizer_file}")
        else:
            print("🗑️ Удаляем optimizer.pt для экономии места")
            optimizer_file.unlink()

    # Copy generated captions to the output tar
    # But do not upload publicly to HF
    captions_dir = JOB_DIR / "captions"
    captions_dir.mkdir(exist_ok=True)
    for caption_file in INPUT_DIR.glob("*.txt"):
        shutil.copy(caption_file, captions_dir)

    os.system(f"tar -cvf {output_path} {JOB_DIR}")

    # Итоговый отчет об обучении
    final_lora_size = (JOB_DIR / "lora.safetensors").stat().st_size / (1024 * 1024)  # MB
    if pretrained_lora_url:
        print("\n🎉 Дообучение LoRA завершено успешно!")
        print(f"   📁 Исходная LoRA: {pretrained_lora_url}")
        print(f"   ✨ Результирующая LoRA: {final_lora_size:.1f} MB")
        print(f"   🎯 Триггер-слово: {trigger_word}")
        print(f"   📊 Выполнено шагов: {steps}")
    else:
        print(f"\n🎉 Обучение LoRA с нуля завершено успешно!")
        print(f"   ✨ Размер LoRA: {final_lora_size:.1f} MB")
        print(f"   🎯 Триггер-слово: {trigger_word}")
        print(f"   📊 Выполнено шагов: {steps}")

    if hf_token is not None and hf_repo_id is not None:
        if captions_dir.exists():
            shutil.rmtree(captions_dir)

        try:
            handle_hf_readme(hf_repo_id, trigger_word, steps, learning_rate, lora_rank, pretrained_lora_url)
            print(f"Uploading to Hugging Face: {hf_repo_id}")
            api = HfApi()

            repo_url = api.create_repo(
                hf_repo_id,
                private=False,
                exist_ok=True,
                token=hf_token.get_secret_value(),
            )

            print(f"HF Repo URL: {repo_url}")

            api.upload_folder(
                repo_id=hf_repo_id,
                folder_path=JOB_DIR,
                repo_type="model",
                use_auth_token=hf_token.get_secret_value(),
            )
        except Exception as e:
            print(f"Error uploading to Hugging Face: {str(e)}")

    return TrainingOutput(weights=Path(output_path))


def handle_hf_readme(
    hf_repo_id: str,
    trigger_word: Optional[str],
    steps: int,
    learning_rate: float,
    lora_rank: int,
    pretrained_lora_url: Optional[str] = None,
):
    readme_path = JOB_DIR / "README.md"
    readme_template_path = Path("hugging-face-readme-template.md")
    shutil.copy(readme_template_path, readme_path)

    with readme_template_path.open() as file:
        template = file.read()

    # Определяем режим обучения для README
    training_mode = "Fine-tuned from pre-trained LoRA" if pretrained_lora_url else "Trained from scratch"
    
    base_details = f"- Training mode: {training_mode}\n- Steps: {steps}\n- Learning rate: {learning_rate}\n- LoRA rank: {lora_rank}"
    
    if pretrained_lora_url:
        base_details += f"\n- Pre-trained LoRA source: {pretrained_lora_url}"
    
    variables = {
        "repo_id": hf_repo_id,
        "title": hf_repo_id.split("/")[1].replace("-", " ").title()
        if len(hf_repo_id.split("/")) > 1
        else hf_repo_id,
        "trigger_word": trigger_word,
        "trigger_section": f"\n## Trigger words\n\nYou should use `{trigger_word}` to trigger the image generation.\n"
        if trigger_word
        else "",
        "instance_prompt": f"instance_prompt: {trigger_word}" if trigger_word else "",
        "training_details": f"\n## Training details\n\n{base_details}\n",
    }

    with readme_path.open("w") as file:
        file.write(Template(template).substitute(variables))


def extract_zip(input_images: Path, input_dir: Path):
    if not is_zipfile(input_images):
        raise ValueError("input_images must be a zip file")

    input_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    with ZipFile(input_images, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.filename.startswith(
                "__MACOSX/"
            ) and not file_info.filename.startswith("._"):
                zip_ref.extract(file_info, input_dir)
                image_count += 1

    print(f"Extracted {image_count} files from zip to {input_dir}")


def clean_up():
    logout_wandb()

    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def download_huggingface_lora(hf_lora_url: str, output_path: str):
    if (
        not hf_lora_url.startswith("https://huggingface.co")
        or ".safetensors" not in hf_lora_url
    ):
        raise ValueError(
            "Invalid URL. Use a HuggingFace download URL like https://huggingface.co/fofr/flux-80s-cyberpunk/resolve/main/lora.safetensors"
        )

    lora_path = OUTPUT_DIR / "flux_train_replicate" / "lora.safetensors"
    print(f"Downloading {hf_lora_url} to {lora_path}")
    subprocess.check_output(
        [
            "pget",
            "-f",
            hf_lora_url,
            lora_path,
        ]
    )
    os.system(f"tar -cvf {output_path} {lora_path}")


def download_pretrained_lora_for_training(pretrained_lora_url: str) -> Path:
    """Загружает готовую LoRA для дообучения"""
    if (
        not pretrained_lora_url.startswith("https://huggingface.co")
        or ".safetensors" not in pretrained_lora_url
    ):
        raise ValueError(
            "Invalid URL. Use a HuggingFace download URL like https://huggingface.co/user/model/resolve/main/lora.safetensors"
        )
    
    # Создаем папку для предварительно обученной LoRA
    pretrained_dir = INPUT_DIR / "pretrained_lora"
    pretrained_dir.mkdir(exist_ok=True, parents=True)
    pretrained_lora_path = pretrained_dir / "pretrained_lora.safetensors"
    
    print(f"Downloading pretrained LoRA from: {pretrained_lora_url}")
    print(f"Saving to: {pretrained_lora_path}")
    
    subprocess.check_output([
        "pget", "-f", pretrained_lora_url, str(pretrained_lora_path)
    ])
    
    if not pretrained_lora_path.exists():
        raise RuntimeError(f"Failed to download pretrained LoRA to {pretrained_lora_path}")
    
    print(f"Successfully downloaded pretrained LoRA ({pretrained_lora_path.stat().st_size} bytes)")
    return pretrained_lora_path


def download_weights():
    if not WEIGHTS_PATH.exists():
        t1 = time.time()
        subprocess.check_output(
            [
                "pget",
                "-xf",
                "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar",
                str(WEIGHTS_PATH.parent),
            ]
        )
        t2 = time.time()
        print(f"Downloaded base weights in {t2 - t1} seconds")
