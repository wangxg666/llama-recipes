import os

import fire
import numpy as np
import tqdm

from utils_ import *
from sklearn.metrics import classification_report, accuracy_score


def main(model_name: str='',
         input_data_file: str='',
         output_model_dir: str='',
         batch_size: int=64,
         evaluation_steps: int=100,
         epochs: int=10,
         valid_split: float=0.05,
         learning_rate: float=1e-4,
         warmup_steps: int=50,
         wandb_name: str=''
         ):

    wandb_writer = WanDBWriter(name=wandb_name, rank=0)

    os.makedirs(output_model_dir, exist_ok=True)

    text_list, label_list = load_data(input_data_file)

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    input_ids = encode_fn(text_list, tokenizer)
    label_ids = torch.tensor(label_list)

    # Split data into train and validation
    dataset = TensorDataset(input_ids, label_ids)
    valid_size = int(valid_split * len(dataset))
    train_size = len(dataset) - valid_size
    train_dataset, val_dataset = random_split(dataset, [train_size, valid_size])

    # Create train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # Load the pretrained BERT model
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2index),
        output_attentions=False,
        output_hidden_states=False)
    model.cuda()

    # create optimizer and learning rate schedule
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    def evaluation():
        model.eval()
        val_losses = []
        real_labels, pred_labels = [], []
        for i, batch in tqdm.tqdm(enumerate(val_dataloader)):
            with torch.no_grad():
                out = model(batch[0].to(device),
                            token_type_ids=None,
                            attention_mask=(batch[0] > 0).to(device),
                            labels=batch[1].to(device))
                loss = out.loss
                logits = out.logits
                val_losses.append(loss.item())
                logits = logits.detach().cpu().numpy()

                pred_labels.extend(np.argmax(logits, axis=1).tolist())
                real_labels.extend(batch[1].to('cpu').numpy().tolist())

        print(real_labels[0:10], pred_labels[0:10])
        # print(classification_report(real_labels, pred_labels, target_names=labels))
        print(f'valid loss is {round(np.mean(val_losses), 5)}', flush=True)
        wandb_writer.log(0, info={
            'valid_loss': round(np.mean(val_losses), 5),
            'valid_accuracy': accuracy_score(real_labels, pred_labels)
        })
        model.train()

    losses = []
    num_steps = 0
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            out = model(batch[0].to(device),
                        token_type_ids=None,
                        attention_mask=(batch[0] > 0).to(device),
                        labels=batch[1].to(device))
            loss = out.loss
            losses.append(loss.item())
            print(f'epoch {epoch}, step {step}, '
                  f'loss = {round(losses[-1], 5)}, '
                  f'loss-smooth = {round(np.mean(losses[-50:]), 5)}', flush=True)

            wandb_writer.log(0, info={
                'step': num_steps,
                'step_loss': loss.detach().float(),
                'learning_rate': scheduler.get_lr()[0]
            })

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            num_steps += 1

            if num_steps % evaluation_steps == 0:
                evaluation()

        evaluation()
        torch.save(model, f'{output_model_dir}/model.epoch.{epoch}.bin')


if __name__ == "__main__":
    fire.Fire(main)
