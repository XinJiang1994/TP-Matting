import sys
sys.path.append("..")
import onnxruntime as ort
import numpy as np
import cv2
from jittor_matter import config
from jittor_matter.models.factory import create_filter,create_model
import time 
import jittor.transform as transforms
from PIL import Image
import jittor as jt

class Matter():
    def __init__(self,im_size=(512,512),onnx_file='assets/model.onnx') -> None:
        self.im_size=im_size
        self.bg_im = None
        self.model=None
        self.crop_size=240
        # self.ort_sess = None
        # self.onnx_file=onnx_file
        # self.model_name=onnx_file.split('/')[-1]
        self.model_name='local'
        self.init_bg()
        self.init_model()
        self.tgt_tensor=None
        self.set_tgt()


    def set_tgt(self,im_path=None):
        # frame_PIL=Image.open('VMDemoSys/assets/tgt_xin2.png')
        frame_PIL=Image.open(im_path)
        
        frame_PIL=frame_PIL.convert('RGB')
        frame_PIL=frame_PIL.resize((512,512))
        frame_tensor = self.jt_transforms(frame_PIL)
        self.tgt_tensor = jt.Var(frame_tensor[None, :, :, :])
        print(self.tgt_tensor.max())
        print(self.tgt_tensor.shape)
        
        self.frame_tensor=jt.zeros([1,3,self.im_size[1],self.im_size[0]])
        self.out_np=np.zeros([1,3,self.im_size[1],self.im_size[0]])

    def init_bg(self):
        r=np.ones(self.im_size)*120
        g=np.ones(self.im_size)*250
        b=np.ones(self.im_size)*150
        self.bg_im=np.array([r,g,b]).transpose([1,2,0])

    def init_model(self):
        # self.ort_sess = ort.InferenceSession(self.onnx_file)
        checkpoint_path = "ckpt_jittor/checkpoint_120.pkl"
        backbone='vit_tiny_patch16_384'
        dataset = 'videomatte'
        decoder = 'mask_transformer'
        dropout=0.0
        drop_path =0

        cfg = config.load_config()
        model_cfg = cfg["model"][backbone]
        dataset_cfg = cfg["dataset"][dataset]
        if "mask_transformer" in decoder:
            decoder_cfg = cfg["decoder"]["mask_transformer"]
        else:
            decoder_cfg = cfg["decoder"][decoder]
        if not self.im_size:
            self.im_size = dataset_cfg["im_size"]
        if not self.crop_size:
            self.crop_size = dataset_cfg.get("crop_size", self.im_size)

        model_cfg["image_size"] = (self.crop_size, self.crop_size)
        model_cfg["backbone"] = backbone
        model_cfg["dropout"] = dropout
        model_cfg["drop_path_rate"] = drop_path
        decoder_cfg["name"] = decoder
        model_cfg["decoder"] = decoder_cfg
        model_cfg["n_cls"] = 16
        tgt_filter = create_filter(model_cfg)
        model=create_model(tgt_filter,'pretrained/tgt_filter.pth','pretrained/matting_model.pth',load_matter=False)
        # model=model.cuda()

        print(f"###################Resuming training from checkpoint: {checkpoint_path}")
        
        ckpt = jt.load(checkpoint_path)
        model.load_state_dict(ckpt['model'])
        self.model=model.eval()

        self.jt_transforms = transforms.Compose(
             [
                transforms.Resize((512,512)),
                transforms.ToTensor(),
            ]
)
    
    def run(self,frame):
        
        out= None
        if self.model_name=='modnet_photographic_portrait_matting.onnx':
            out= self.run_modnet(frame)
        elif self.model_name=='model.onnx':
            out= self.run_ours(frame)
        else:
            out=self.run_local(frame)
        return out
    
    def run_ours(self,frame,):
        H,W = frame.shape[:2]
        frame=cv2.resize(frame,self.im_size)
        x=np.concatenate([frame,frame],axis=2).transpose([2,0,1]).astype(np.float32)[np.newaxis,:,:,:]
        outputs = self.ort_sess.run(None, {'xxt':x})
        # print(len(outputs)) # 6
        # print(outputs[0].shape) # (1, 1, 512, 512)
        out=outputs[0][0].transpose([1,2,0]).repeat(3,axis=2)
        frame_new=(out*255).astype(np.uint8)
        
        # frame_new=frame*pha+self.bg_im*(1-pha)
        frame_new = cv2.resize(frame_new,(W,H))
        return frame_new
    
    def run_modnet(self,frame):
        H,W = frame.shape[:2]
        frame=cv2.resize(frame,self.im_size)
        # x=np.concatenate([frame,frame],axis=2).transpose([2,0,1]).astype(np.float32)[np.newaxis,:,:,:]
        x=frame.transpose([2,0,1])[np.newaxis,:,:,:].astype(np.float32)
        outputs = self.ort_sess.run(None, {'input':x})
        # print(len(outputs)) # 1
        # print(outputs[0].shape) # (1, 1, 512, 512)
        out=outputs[0][0].transpose([1,2,0]).repeat(3,axis=2)
        frame_new=(out*255).astype(np.uint8)
        
        # frame_new=frame*pha+self.bg_im*(1-pha)
        frame_new = cv2.resize(frame_new,(W,H))
        return frame_new
    
    def run_local(self,frame):
        st=time.time()
        H,W = frame.shape[:2]
        print('####### frame.shape: ',frame.shape)
        frame_PIL = Image.fromarray(frame)
        to_im=time.time()
        print('@to im cost: ',to_im-st)

        self.frame_tensor = self.jt_transforms(frame_PIL)
        self.frame_tensor = jt.Var(self.frame_tensor[None, :, :, :])
        to_Var=time.time()
        print('@to to_Var cost: ',to_Var-to_im)
        
        with jt.no_grad():
            out = self.model(self.frame_tensor, self.tgt_tensor,inference=True)
            out[0].sync()
        
        time_infer=time.time()
        print('@time_infer cost: ',time_infer-to_Var)

        self.out_np=out[0][0].data # C H W
        # print('### type(self.out_np): ',self.out_np)
        pha=self.out_np
        to_np=time.time()
        print('@to_np cost: ',to_np-time_infer)
        pha=pha.repeat(3,axis=0).transpose(1,2,0)
        pha=cv2.normalize(pha,None,0,255,cv2.NORM_MINMAX)
        ret=pha.astype(np.uint8)
        # print('#####: ',ret.shape)
        frame_new = cv2.resize(ret,(W,H))
        np_process=time.time()
        print('@np_process cost: ',np_process-to_np)

        print('## totaltime cost: ',time.time()-st)
        
        return frame_new
