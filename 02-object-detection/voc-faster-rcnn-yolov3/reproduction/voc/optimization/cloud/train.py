import json, os, subprocess, sys, time, urllib.request, hashlib
from pathlib import Path

ROOT=Path('/kaggle/working')
REPO=ROOT/'computer-vision-projects'
if not REPO.exists():
    subprocess.run(['git','clone','--depth','1','--branch','portfolio/reproductions-2026-11-23','https://github.com/fcxuu/computer-vision-projects.git',str(REPO)],check=True)
module=REPO/'02-object-detection/voc-faster-rcnn-yolov3/reproduction/voc'
sys.path.insert(0,str(module))
from voc import load_split, VocDataset
from detectors import Detector, original_detections
from experiment import stack, assess, loader, save_checkpoint
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

def download():
    root=ROOT/'voc'
    root.mkdir(exist_ok=True)
    for name,md5 in [('VOCtrainval_06-Nov-2007.tar','c52e279531787c972589f7e41ab4ae64'),('VOCtest_06-Nov-2007.tar','b6e924de25625d8de591ea690078ad9f')]:
        file=root/name
        if not file.exists():
            urllib.request.urlretrieve('https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/'+name,file)
        with file.open('rb') as f:
            assert hashlib.file_digest(f,'md5').hexdigest()==md5
    for split in ('train','val','test'):
        print(split,len(load_split(root,split)),flush=True)

def worker():
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    device=torch.device('cuda',rank)
    dist.init_process_group('nccl')
    torch.manual_seed(2020)
    root=ROOT/'voc'
    train=load_split(root,'train')
    detector=Detector('faster_rcnn',320,device,ROOT/'vendor',root,pretrained=True)
    model=DDP(detector.net,device_ids=[rank],broadcast_buffers=False)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.003,momentum=.9,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,8,eta_min=.00015)
    dataset=VocDataset(train,320)
    sampler=DistributedSampler(dataset,num_replicas=2,rank=rank,shuffle=True,seed=2020)
    batches=DataLoader(dataset,batch_size=8,sampler=sampler,collate_fn=stack,num_workers=2)
    record=ROOT/'faster-results'
    if rank==0: record.mkdir(exist_ok=True)
    best=-1
    history=[]
    for epoch in range(1,9):
        sampler.set_epoch(epoch)
        detector.training_mode()
        total=torch.zeros(2,device=device)
        for pixels,targets in batches:
            images=pixels.to(device,dtype=torch.float32)/255
            ground=detector.targets(targets)
            optimizer.zero_grad(set_to_none=True)
            losses=model(list(images),ground)
            loss=sum(losses.values())
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            optimizer.step()
            total+=torch.stack((loss.detach()*len(targets),torch.tensor(len(targets),device=device)))
        dist.all_reduce(total)
        scheduler.step()
        dist.barrier()
        if rank==0:
            val=load_split(root,'val')
            metrics,_=assess(detector,val,loader(val,320,8),include_loss=True)
            row={'epoch':epoch,'train_loss':float(total[0]/total[1]),'val_loss':metrics['loss'],'val_map50':metrics['map50'],'gpu0_peak_bytes':torch.cuda.max_memory_allocated(0)}
            history.append(row)
            print('EPOCH',json.dumps(row),flush=True)
            if metrics['map50']>best:
                best=metrics['map50']
                save_checkpoint(record/'best.pt',detector,epoch)
                (record/'best-val.json').write_text(json.dumps(metrics,indent=2))
            (record/'history.json').write_text(json.dumps(history,indent=2))
        peak=torch.tensor(torch.cuda.max_memory_allocated(rank),device=device,dtype=torch.int64)
        peaks=[torch.zeros_like(peak) for _ in range(2)]
        dist.all_gather(peaks,peak)
        if rank==0:
            print('GPU_PEAK_BYTES', [int(x) for x in peaks],flush=True)
        dist.barrier()
    dist.destroy_process_group()

if __name__=='__main__':
    if 'LOCAL_RANK' in os.environ:
        worker()
    else:
        print(subprocess.check_output(['nvidia-smi'],text=True),flush=True)
        assert torch.cuda.device_count()==2
        download()
        subprocess.run(['torchrun','--standalone','--nproc-per-node=2',__file__],check=True)
        device=torch.device('cuda:0')
        from experiment import load_checkpoint
        detector,state=load_checkpoint(ROOT/'faster-results/best.pt',device,ROOT/'vendor',ROOT/'voc')
        test=load_split(ROOT/'voc','test')
        metrics,_=assess(detector,test,loader(test,320,8),include_loss=False)
        metrics.update(variant='faster_rcnn',epoch=state['epoch'],size=320)
        (ROOT/'faster-results/test.json').write_text(json.dumps(metrics,indent=2))
        print('TEST',json.dumps(metrics),flush=True)
