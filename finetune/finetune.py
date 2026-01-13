import time
import torch
import torch.nn as nn
from dataset import Dataset, custom_collate
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from transformers import get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

def validation_step(model, validation_loader, save_dir, tokenizer):
    print('start validation')
    validation_loss = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            data = {
                'input_ids': data['input_ids'].to(device_ids[0]),
                'labels': data['labels'].to(device_ids[0]),
                'attention_mask': data['attention_mask'].to(device_ids[0])
            }
            output = model(input_ids=data['input_ids'], labels=data['labels'], attention_mask=data['attention_mask'],
                           return_dict=True)
            loss = output.loss
            validation_loss.append(loss.mean().item())
    print('validation loss:', round(sum(validation_loss) / len(validation_loss), 4))
    model.module.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    model.train()


def fine_tune(training_file, validation_file, epochs, batch_size, save_dir, load_range=None):
    tokenizer = RobertaTokenizer.from_pretrained('./CodeT5')
    base_model = T5ForConditionalGeneration.from_pretrained('CodeT5')
    model = get_peft_model(base_model, peft_config)

    print('model parameters:', sum(param.numel() for param in model.parameters()))
    model = nn.DataParallel(model, device_ids=device_ids).to(device_ids[0])

    training_dataset = Dataset(training_file, tokenizer, max_length=512, shuffle=False, load_range=load_range)
    validation_dataset = Dataset(validation_file, tokenizer, max_length=512, load_range=None)
    training_sampler = torch.utils.data.SequentialSampler(training_dataset)
    validation_sampler = torch.utils.data.SequentialSampler(validation_dataset)
    training_loader = torch.utils.data.DataLoader(
        dataset=training_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=training_sampler, collate_fn=custom_collate
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=2 * batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=validation_sampler, collate_fn=custom_collate
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=0, num_training_steps=int(epochs * len(training_loader))
    )
    for epoch in range(epochs):
        model.train()
        training_loss = []
        start_time = time.time()
        oom = 0
        for i, data in enumerate(training_loader):
            data = {
                'input_ids': data['input_ids'].to(device_ids[0]),
                'labels': data['labels'].to(device_ids[0]),
                'attention_mask': data['attention_mask'].to(device_ids[0])
            }
            try:
                optimizer.zero_grad()
                output = model(input_ids=data['input_ids'], labels=data['labels'],
                               attention_mask=data['attention_mask'], return_dict=True)
                loss = output.loss

                loss.mean().backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.3)
                optimizer.step()
                scheduler.step()
                training_loss.append(loss.mean().item())
            except Exception as e:
                if 'out of memory' in str(e):
                    oom += 1
                model.zero_grad()
                optimizer.zero_grad()
                del data

                torch.cuda.empty_cache()

            if i % 1000 == 0:
                print('epoch: {}, step: {}/{}, loss: {}, lr: {}, oom: {}, time: {}s'.format(
                    epoch + 1, i, len(training_loader),
                    round(sum(training_loss) / len(training_loss), 4),
                    round(scheduler.get_last_lr()[0], 6), oom,
                    int(time.time() - start_time)
                ))
                start_time = time.time()
                oom = 0
            if i % 10000 == 0 and i > 0:
                validation_step(model, validation_loader, save_dir, tokenizer)
    validation_step(model, validation_loader, save_dir, tokenizer)


if __name__ == '__main__':
    device_ids = [0]
    training_file = 'data/finetune_training.jsonl'  # change to fine-tuning data path
    validation_file = 'data/finetune_validation.jsonl'  # change to fine-tuning data path
    vocabulary_file = 'codet5-large'
    pretrained_file = 'codet5-large'

    fine_tune(
        training_file, validation_file, epochs=5, batch_size=1, save_dir='codet5-large-finetune_ast/', load_range=None
    )
