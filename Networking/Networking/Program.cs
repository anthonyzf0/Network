using System;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.ML.Train;
using Encog.ML.Data.Basic;
using Encog;
using System.IO;
using System.Text;
using System.Linq;
using System.Collections.Generic;

namespace Networking
{
    class Program
    {

        static string e = File.ReadAllText(@"data\english.txt", Encoding.UTF8);
        static string f = File.ReadAllText(@"data\french.txt", Encoding.UTF8);

        private static void Main(string[] args)
        {
            List<Double[]> InData = new List<double[]>();
            List<Double[]> OutData = new List<double[]>();

            //Does english values
            double[][] english = null;
            english = e.Replace("\n", "|").Replace("\r", "").Split('|').ToList().Where(i => i.Length == 5).Select(i => i.ToCharArray().ToList().Select(j => ((double)(Convert.ToInt32(j) - 97)) / 26).ToArray()).ToArray();

            List<double[]> eng = english.ToList();
            eng.RemoveRange(433, english.Count() - 433);
            english = eng.ToArray();

            foreach (double[] d in english)
            {
                InData.Add(d);
                OutData.Add(new Double[] { 1 });
            }

            //French values
            Char[] chars = ".1234567890 \r".ToCharArray();
            foreach (char c in chars)
                f = f.Replace(c + "", "");

            List<String> values = f.Replace("\n", "|").Split('|').ToList();
            for (int i = values.Count - 1; i >= 0; i--)
            {
                if (values[i].Length != 5)
                {
                    values.RemoveAt(i);
                    continue;
                }

                bool remove = false;

                foreach (char c in values[i].ToCharArray())
                    if ((int)c < 97 || (int)c > 123)
                        remove = true;
                if (remove)
                    values.RemoveAt(i);
            }
            double[][] french = values.Select(i => i.ToCharArray().ToList().Select(j => ((double)(Convert.ToInt32(j) - 97)) / 26).ToArray()).ToArray();

            foreach (double[] d in french)
            {
                InData.Add(d);
                OutData.Add(new Double[] { 0 });
            }

            Console.WriteLine("English : " + english.GetLength(0) + " French : " + french.GetLength(0));

            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(null, true, 5));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 26));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 13));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 8));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
            network.Structure.FinalizeStructure();
            network.Reset();

            IMLDataSet trainingSet = new BasicMLDataSet(InData.ToArray(), OutData.ToArray());

            IMLTrain train = new ResilientPropagation(network, trainingSet);

            int epoch = 1;

            do
            {
                train.Iteration();
                Console.WriteLine(@"Epoch #" + epoch + @" Error:" + train.Error);
                epoch++;
            } while (train.Error > 0.24);

            train.FinishTraining();

            Console.WriteLine("______________");
            double[] outputs = new double[] {1 };
            while (true)
            {
                String word = Console.ReadLine();
                if (word.Length != 5) continue;

                double[] data = word.ToCharArray().Select(i => ((double)Convert.ToInt32(i) - 97) / 26).ToArray();


                network.Compute(data, outputs);
                int num = (int)Math.Round(outputs[0]);

                Console.WriteLine((num == 1)?"English":"French");
                Console.WriteLine("Certainty : " + Math.Round((1-Math.Abs(1 - outputs[0])),2)*100+" %");
                Console.WriteLine(outputs[0]);
               
            }
        }
            
    }  
}
