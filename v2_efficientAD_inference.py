import torch
import cv2
import time
import numpy as np
from torchvision import transforms
from PIL import Image
from v2_efficientAD import predict  # pastikan fungsi predict dapat diimpor dari file trainingmu

# === Parameter ===
#img_path = 'ingot\\test\\benda_asing\\cropped_rect_20250722_152148_jpg.rf.299626f446e010790de815f62e46c5ab.jpg'  # ganti dengan path gambar
img_path = 'ingot\\train\\good\\6.jpg'  # ganti dengan path gambar
path = 'output/2/trainings/'
teacher_path = f'{path}teacher_final.pth'
teacher_mean_path = f'{path}teacher_mean.pth'
teacher_std_path = f'{path}teacher_std.pth'
student_path = f'{path}student_final.pth'
autoencoder_path = f'{path}autoencoder_final.pth'
image_size = 256
out_channels = 384
on_gpu = torch.cuda.is_available()

teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
student = torch.load(student_path, map_location='cpu', weights_only=False)
autoencoder = torch.load(autoencoder_path, map_location='cpu', weights_only=False)
teacher_mean = torch.load(teacher_mean_path, map_location='cpu')
teacher_std = torch.load(teacher_std_path, map_location='cpu')

teacher.eval()
student.eval()
autoencoder.eval()

if on_gpu:
    teacher.cuda()
    student.cuda()
    autoencoder.cuda()
    teacher_mean = teacher_mean.cuda()
    teacher_std = teacher_std.cuda()

# === Preprocess image ===
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
orig_img = Image.open(img_path).convert('RGB')
orig_width, orig_height = orig_img.size
input_tensor = default_transform(orig_img).unsqueeze(0)

if on_gpu:
    input_tensor = input_tensor.cuda()

start_time = time.time()

# === Infer satu gambar ===
map_combined, map_st, map_ae = predict(
    image=input_tensor,
    teacher=teacher,
    student=student,
    autoencoder=autoencoder,
    teacher_mean=teacher_mean,
    teacher_std=teacher_std,
    q_st_start=None,
    q_st_end=None,
    q_ae_start=None,
    q_ae_end=None,
)

# === Resize ke ukuran asli ===
map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
map_combined = torch.nn.functional.interpolate(map_combined, (orig_height, orig_width), mode='bilinear')
heatmap_np = map_combined[0, 0].cpu().numpy()
score_max = np.max(heatmap_np)
score_mean = np.mean(heatmap_np)
print(score_max, score_mean)

# === Normalisasi heatmap ke 0-255 dan warnai ===  
heatmap_norm = cv2.normalize(heatmap_np, None, 0, 255, cv2.NORM_MINMAX)
heatmap_uint8 = heatmap_norm.astype(np.uint8)
colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

# === Gabungkan dengan gambar asli ===
orig_img_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
overlay = cv2.addWeighted(orig_img_cv, 0.5, colored_heatmap, 0.5, 0)
finish_time = time.time()
total_time = round(finish_time-start_time, 2)
print(total_time)
# === Simpan hasil overlay ===
cv2.imwrite('overlay_result.jpg', overlay)
print("Overlay disimpan ke overlay_result.jpg")
cv2.imshow('Overlay result', overlay)
cv2.waitKey()