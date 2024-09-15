import numpy as np
from skimage.metrics import structural_similarity
from scipy.spatial.distance import cosine
def cos(image1,image2):
    vector1 = image1.flatten()  # 将第一张图像转换为一维向量
    vector2 = image2.flatten()  # 将第二张图像转换为一维向量

    similarity = 1 - cosine(vector1, vector2)
    return similarity,cosine(vector1, vector2)
def zero_out_mask_area(image, mask, radius):
    """
    Zero out the pixels in the surrounding mask area of each pixel.

    Args:
        image (ndarray): Input image.
        mask (ndarray): Mask image where 1 indicates the area to keep and 0 indicates the surrounding area.
        radius (int): Radius of the surrounding mask area.

    Returns:
        ndarray: Processed image with the surrounding mask area zeroed out.
    """
    # Create a copy of the input image
    processed_image = np.copy(image)

    # Get the coordinates of pixels where the mask is 0 (surrounding area)

    h, w = image.shape[:2]
    max_row = h -  radius
    max_col = w -  radius

    # Generate random starting coordinates
    start_row = np.random.randint(0, max_row)
    start_col = np.random.randint(0, max_col)
    # Iterate over the surrounding pixels and set them to zero

    processed_image[start_row:start_row + radius + 1, start_col :start_col + radius + 1] = 0

    return processed_image

def calculate_psnr(original_img, reconstructed_img):
    # 将图像数据转换为浮点类型
    original_img = original_img.astype(np.float64)
    reconstructed_img = reconstructed_img.astype(np.float64)

    # 计算图像的均方误差（Mean Squared Error，MSE）
    mse = np.mean((original_img - reconstructed_img) ** 2)

    # 计算最大可能的像素值
    max_pixel = np.max(original_img)

    # 计算PSNR
    psnr = 10 * np.log10((max_pixel ** 2) / mse)

    return psnr
def calculate_similarity(original_image, current_image):
    """
    Calculate the similarity between the original and current features using structural similarity index (SSIM).

    Args:
        original_image (ndarray): Original image.
        current_image (ndarray): Current image.

    Returns:
        float: Similarity score between the two images.
    """
    similarity = structural_similarity(original_image, current_image, multichannel=True)
    return similarity
import matplotlib.pyplot as plt
from PIL import Image
# Example usage  imgA = np.asarray(imgA)
original_image =   np.asarray(Image.open(r'C:\Users\Dell\Desktop\Image_Net\红外\MMIF-CDDFuse-main\MMIF-CDDFuse-main\D\D.jpg') ) # Placeholder for original image
current_image =  np.asarray(Image.open(r'D:\Image_Data\IRVI\AUIF Datasets\16x\Test_TNO\VIS22.bmp'))  # Placeholder for current image
h,w = original_image.shape
mask = np.ones((h,w))  # Placeholder for mask image

radius = 0  # Radius of the surrounding mask area

# Zero out the surrounding mask area in the current image
processed_image = zero_out_mask_area(current_image, mask, radius)
# plt.figure("Image") # 图像窗口名称
# plt.imshow(processed_image)
# plt.axis('on') # 关掉坐标轴为 off
# plt.title('image') # 图像题目
# plt.show()
# Calculate the similarity between the original and processed images
similarity_score = calculate_similarity( current_image,original_image)
PSNR= calculate_psnr( current_image,original_image)
print("Similarity Score:", similarity_score)
print("PSNR Score:", PSNR)
