using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace AiHelper
{
    public class Yolo
    {
        /// <summary>
        /// 预处理图片
        /// </summary>
        /// <param name="imgStream">图片的流</param>
        /// <param name="targetWidth">转换后的大小</param>
        /// <param name="targetHeight">转换后的大小</param>
        /// <returns></returns>
        public static Tensor<float> PreprocessImage(Stream imgStream,int targetWidth=640,int targetHeight=640)
        {
            // 使用SkiaSharp进行图像处理
            using (SKBitmap skBitmap = SKBitmap.Decode(imgStream))
            using (SKBitmap resizedBitmap = skBitmap.Resize(new SKImageInfo(targetWidth, targetHeight), SKFilterQuality.High))
            {
                // 将图片像素转换为浮点数数组，存储为 [channels, width, height]
                float[] imageData = new float[3 * targetWidth * targetHeight]; // 3是因为RGB三通道
                int indexR = 0;
                int indexG = targetWidth * targetHeight;
                int indexB = 2 * targetWidth * targetHeight;
                for (int y = 0; y < resizedBitmap.Height; y++)
                {
                    for (int x = 0; x < resizedBitmap.Width; x++)
                    {
                        SKColor pixel = resizedBitmap.GetPixel(x, y);
                        // 将像素值归一化到0-1之间
                        imageData[indexR++] = pixel.Red / 255.0f;
                        imageData[indexG++] = pixel.Green / 255.0f;
                        imageData[indexB++] = pixel.Blue / 255.0f;
                    }
                }
                // 将数据转换为Tensor<float>
                var dimensions = new[] { 1, 3, targetHeight, targetWidth }; // batch size 为 1, 通道在前
                return new DenseTensor<float>(imageData, dimensions);
            }
        }
    }
}