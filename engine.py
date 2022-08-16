from tqdm import tqdm
import torch
import config

def train_fn(model, dataloader, optimizer):
    model.train()
    final_loss = 0
    tk = tqdm(dataloader, total = len(dataloader))
    for data in tk:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        optimizer.zero_grad()
        # _, loss = model(**data)
        _, loss = model(data["images"], data["labels"])
        loss.backward()
        optimizer.step()
        final_loss += loss.item()
                
        # del data
        # torch.cuda.empty_cache()

    return final_loss/len(dataloader)


def eval_fn(model, dataloader):
    model.eval()
    final_loss = 0
    final_preds = []
    tk = tqdm(dataloader, total = len(dataloader))
    for data in tk:
        with torch.no_grad():
            for k, v in data.items():
                data[k] = v.to(config.DEVICE)

            batch_preds, loss = model(data["images"], data["labels"])
            final_loss += loss.item()
            final_preds.append(batch_preds)

        # del data
        # torch.cuda.empty_cache()

    return final_preds, final_loss/len(dataloader)