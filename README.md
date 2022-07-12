# DebCM: Debiased Cross-modal Matching for Content-based Micro-video Background Music Recommendation
This is our implementation of DebCM for recommendation systems associated with:

>**DebCM: Debiased Cross-modal Matching for Content-based Micro-video Background Music Recommendation,**  
>Jing Yi and Zhenzhong Chen  
 
## Environment Requirement
- python == 3.7
- Pytorch == 1.4.0
## Directory Structure
- Model_teacher.py: Model of the teacher network.
- DataLoad_teacher.py: DataLoader of the teacher network.
- train_teacher_net.py: Training script of the teacher network.
- Model.py: Model of the student network.
- DataLoad.py: DataLoader of the student network.
- train_with_pgc_distillation.py: Training script of the student network which utilizes the knowledge of teacher model.
- evaluate.py: Evaluation script of the student network on test set. 
## Dataset Format
- visual_features_resnet_V.npy: video features of Nx1024 extracted from ResNet.
- music_features.npy: music features of Nx128 extracted from Vggish.
- new_music_smile_features_norm.py: music features of Nx128 extracted from OpenSmile.
- train.npy: Train file. Each line is [a video, the positive background music, genre of the video, music genres distributions of the uploader as user_prior, propensity for IPS] (videos and music are indexed as videoID and background music ID).
- 100neg/val.npy: Validation file. Each line is [a video, the positive music, array of several sampled negative music].
- 100neg/test.npy: Test file. Each line is [a video, the positive music, array of several sampled negative music].
- video_popularity_val.npy: Inversed-popularity weight of videos associated to their matched music in validation set.
- video_popularity_test.npy: Inversed-popularity weight of videos associated to their matched music in test set.
- user_prior_mean.py: Average of music genres distributions of uploaders in the training set.

### **If you find our codes helpful, please kindly cite the following paper. Thanks!**
	@article{debcm,
	  title={Multi-modal Variational Graph Auto-encoder for Recommendation Systems},
	  author={Yi, Jing and Chen, Zhenzhong},
	  year={2022},
	}
