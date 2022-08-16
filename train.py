from cgi import test
import torch
import numpy as np
from torch.utils.data import DataLoader

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
import torchvision
from torchvision import models
import config
import dataset
import pandas as pd
import label_encoded
from model import TextRecognition
import engine
import label_encoded

def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k-1
            if k == -1:
                temp.append("*")
            else:
                temp.append(encoder.inverse_transform([k+1])[0])
        tp = "".join(temp)

        preds_decoded = ''
        chars = [k[0] for k in tp.split('*') if k != ''] #splitting string based on blank character
        tp = preds_decoded.join(chars)        
        cap_preds.append(tp)
    return cap_preds

def run_training():
    image_files, labels, targets_orig, lbl_encoder = label_encoded.encode_data()
    num_chars = len(lbl_encoder.classes_)

    train_images, test_images, train_targets, test_targets, train_orig_targets, test_orig_targets = model_selection.train_test_split(
        image_files, labels, targets_orig, test_size=0.1, random_state=42)

    train_dataset = dataset.ImageDataset(
        train_images, 
        train_targets, 
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        )
 
    train_loader = DataLoader(train_dataset, batch_size = config.BATCH_SIZE,num_workers = config.NUM_WORKERS, shuffle = True)

    test_dataset = dataset.ImageDataset(
        test_images, 
        test_targets, 
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))

    test_loader = DataLoader(test_dataset,batch_size = config.BATCH_SIZE,num_workers = config.NUM_WORKERS, shuffle = False)

    # Build Model


    model = TextRecognition(num_chars=num_chars)
    optimizer = torch.optim.Adam(model.parameters(),lr = 3e-4)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor = 0.8, patience =5, verbose = True)
    
    checkpoint = torch.load("weights/best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.to(config.DEVICE)


    min_loss = 10000

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.eval_fn(model, test_loader)

        valid_cap_preds = []
        for vp in valid_preds:
            current_preds = decode_predictions(vp, lbl_encoder)
            valid_cap_preds.extend(current_preds)

        #df = pd.DataFrame(list(zip(test_orig_targets, valid_cap_preds)))
        #df.to_csv(f'preds/preds_{epoch}.csv',encoding='utf-8-sig', index = False)
        
        print(list(zip(test_orig_targets, valid_cap_preds))[6:11])

        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Validation loss: {valid_loss}")

        if min_loss > train_loss:
            min_loss = train_loss
            torch.save({ 'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': min_loss,}, "weights/best.pt")

if __name__ == "__main__":
    run_training()