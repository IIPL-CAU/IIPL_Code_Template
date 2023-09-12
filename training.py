# Import modules

# Import PyTorch
import torch
# Import custom modules
from models.model_init import model_init
from utils.data_load import data_load
from utils.optimizer import get_optimizer
from utils.scheduler import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from models.dataset.dataset_init import dataset_init
from torch import nn
from tqdm import tqdm
import wandb

from models.tokenizer.tokenizer_init import tokenizer_load

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Load
    train_src_list, train_trg_list = data_load(dataset_path=args.dataset_path, data_split_ratio=args.data_split_ratio,
                                               seed=args.seed, mode='train')
    valid_src_list, valid_trg_list = data_load(dataset_path=args.dataset_path, data_split_ratio=args.data_split_ratio,
                                               seed=args.seed, mode='valid')
    args.num_classes = len(set(train_trg_list))

    # Model Load
    model = model_init(args)
    model.to(device)
    wandb.watch(model)

    if args.task =='single_text_classification':
        if args.model == "bert-base-uncased":

            # tokenizer init
            src_tokenizer = tokenizer_load(args)

            # Train dataset setting
            custom_dataset_dict = dict()
            custom_dataset_dict['src_tokenizer'] = src_tokenizer
            custom_dataset_dict['src_list'] = train_src_list
            custom_dataset_dict['trg_list'] = train_trg_list
            train_dataset = dataset_init(args=args, dataset_dict=custom_dataset_dict)

            # Valid dataset setting
            custom_dataset_dict['src_list'] = valid_src_list
            custom_dataset_dict['trg_list'] = valid_trg_list
            valid_dataset = dataset_init(args=args, dataset_dict=custom_dataset_dict)
            
            # Dataloader setting
            dataloader_dict = {
                'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    pin_memory=True, num_workers=args.num_workers),
                'valid': DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                    pin_memory=True, num_workers=args.num_workers)
            }

            # Optimizer setting
            optimizer = get_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay, optim_type=args.optim_type)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * args.epochs)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(args.epochs):
                print(f"Epoch {epoch + 1}/{args.epochs}")
                
                for phase in ['train', 'valid']:
                    if phase == 'train':
                        model.train()
                    if phase == 'valid':
                        # write_log(logger, 'Validation start...')
                        model.eval()

                    for batch in tqdm(dataloader_dict[phase]):

                        # Optimizer gradient setting
                        optimizer.zero_grad()

                        # Input setting
                        src_sequence = batch['src_sequence'].to(device)
                        src_attention_mask = batch['src_attention_mask'].to(device)
                        trg_label = batch['trg_label'].to(device)

                        # Model processing
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(input_ids=src_sequence, attention_mask=src_attention_mask)

                        # Loss back-propagation
                        loss = criterion(outputs, trg_label)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()

                        if phase == 'valid':
                            pass
                            # 결과 출력 및 모델 저장 코드 여기 들어가야함
                            # [아래는 예시]
                            # save_file_name = os.path.join(args.model_save_path, args.data_name, args.aug_encoder_model_type, f'checkpoint_seed_{args.random_seed}.pth.tar')
                            # if val_recon_loss < best_aug_val_loss:
                            #     write_log(logger, 'Model checkpoint saving...')
                            #     torch.save({
                            #         'cls_training_done': True,
                            #         'epoch': epoch,
                            #         'model': model.state_dict(),
                            #         'aug_cls_optimizer': aug_cls_optimizer.state_dict(),
                            #         'aug_recon_optimizer': aug_recon_optimizer.state_dict(),
                            #         'aug_cls_scheduler': aug_cls_scheduler.state_dict(),
                            #         'aug_recon_scheduler': aug_recon_scheduler.state_dict(),
                            #     }, save_file_name)
                            #     best_aug_val_loss = val_recon_loss
                            #     best_aug_epoch = epoch
                            # else:
                            #     else_log = f'Still {best_aug_epoch} epoch Loss({round(best_aug_val_loss.item(), 2)}) is better...'
                            #     write_log(logger, else_log)

                    # print(f'Epoch {epoch + 1}/ loss : {loss}')
                    #test 코드에 metric 작성되면 validation코드도 추가  



        # torch.save(model.state_dict(), args.model_path)

    # if args.task =='multi_text_classification':
    #     pass

    # if args.task =='machine_translation':
    #     pass

    # return None