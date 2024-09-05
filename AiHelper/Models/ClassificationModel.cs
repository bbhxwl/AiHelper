namespace AiHelper.Models
{
    public class ClassificationModel
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public float Confidence { get; set; }
        public int LabelIndex { get; set; }
    }
}