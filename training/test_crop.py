import SimpleITK as sitk, numpy as np, glob, logging, time
logging.getLogger('radiomics').setLevel(logging.ERROR)
from radiomics import featureextractor
pid='MDA-420'
mp=glob.glob(f'/N/scratch/skachole/hecktor_nnunet/results/Dataset501_HECKTOR/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_*/validation/{pid}.nii.gz')[0]
ctp=glob.glob(f'/N/project/Sanket_Slate_Project/1_datasets/MICCAI_HECKTOR/HECKTOR 2026 Training Data/{pid}/*CT*.nii.gz')[0]
mimg=sitk.ReadImage(mp); ct=sitk.ReadImage(ctp)
marr=sitk.GetArrayFromImage(mimg)
idx=np.argwhere(marr>0); lo=np.maximum(idx.min(0)-10,0); hi=np.minimum(idx.max(0)+10,np.array(marr.shape))
print('full shape',marr.shape,'-> crop box',(hi-lo).tolist(),flush=True)
sl=(slice(lo[0],hi[0]),slice(lo[1],hi[1]),slice(lo[2],hi[2]))
m_c=marr[sl]; c_c=sitk.GetArrayFromImage(ct)[sl]
mimg_c=sitk.GetImageFromArray(m_c); mimg_c.SetSpacing(mimg.GetSpacing())
ct_c=sitk.GetImageFromArray(c_c); ct_c.CopyInformation(mimg_c)
mb=sitk.GetImageFromArray((m_c==2).astype('uint8')); mb.CopyInformation(mimg_c)
s={'resampledPixelSpacing':[2,2,2],'interpolator':'sitkBSpline','binWidth':25.0,'label':1,'geometryTolerance':1e-3,'padDistance':10}
ex=featureextractor.RadiomicsFeatureExtractor(**s)
t=time.time(); print('extracting...',flush=True)
r=ex.execute(ct_c, mb)
print('DONE',round(time.time()-t,1),'s,',len([k for k in r if not k.startswith('diag')]),'features',flush=True)
