using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiHelper.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace AiHelper
{
    public class Yolo
    {
        InferenceSession _session;
        public Yolo(string onnxPath)
        {
            _session = new InferenceSession(onnxPath);
        }
        /// <summary>
        /// 预测
        /// </summary>
        /// <param name="imgStream"></param>
        /// <param name="confidence"></param>
        /// <returns></returns>
        public List<ClassificationModel> Prediction(Stream imgStream,float confidence=0.5f)
        {
            SKBitmap originalBitmap = SKBitmap.Decode(imgStream);
            int originalWidth = originalBitmap.Width;
            int originalHeight = originalBitmap.Height;
            int inputWidth = 640; 
            int inputHeight = 640;
            List<NamedOnnxValue> inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("images",PreprocessImage(originalBitmap,inputWidth,inputHeight)) };
            
            var results = _session.Run(inputs);
            var output = results.First().AsTensor<float>();
            var boxes = new List<float[]>();
            for (int i = 0; i < output.Dimensions[1]; i++)
            {
                var boxData = new float[85];
                for (int j = 0; j < 85; j++)
                {
                    boxData[j] = output[0, i, j];
                }
                boxes.Add(boxData);
            }
            List<ClassificationModel> classificationModels = new List<ClassificationModel>();
            foreach (var box in boxes)
            {
                float tempConfidence = box[4];
                if (tempConfidence>=confidence)
                {
                    classificationModels.Add(new ClassificationModel
                    {
                        X = (int) box[0],
                        Y = (int) box[1],
                        Width = (int)box[2],
                        Height = (int) box[3],
                        Confidence = tempConfidence,
                        LabelIndex = box.Skip(5).ToList().IndexOf(box.Skip(5).Max())
                    });
                }
            }
            return classificationModels;
        }
        /// <summary>
        /// 预处理图片
        /// </summary>
        /// <param name="imgStream">图片的流</param>
        /// <param name="targetWidth">转换后的大小</param>
        /// <param name="targetHeight">转换后的大小</param>
        /// <returns></returns>
        public static Tensor<float> PreprocessImage(Stream imgStream, int targetWidth = 640, int targetHeight = 640)
        {
            // 使用SkiaSharp进行图像处理
            using (SKBitmap skBitmap = SKBitmap.Decode(imgStream))
            using (SKBitmap resizedBitmap =
                   skBitmap.Resize(new SKImageInfo(targetWidth, targetHeight), SKFilterQuality.High))
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
        public static Tensor<float> PreprocessImage(SKBitmap skBitmap, int targetWidth = 640, int targetHeight = 640)
        {
            using (SKBitmap resizedBitmap =
                   skBitmap.Resize(new SKImageInfo(targetWidth, targetHeight), SKFilterQuality.High))
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