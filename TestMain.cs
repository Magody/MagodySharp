using System;
using System.Collections.Generic;

using layers;
using types;
using custom_lib;


namespace magodysharp
{
    public struct DataFormat
    {
        public MatrixD X;
        public MatrixD y;
    }
    class TestMain
    {
        
        public static void Main(string[] args)
        {


            TestDQNNMnistDigits.tryTest();

            /*           
            
            // Testing neural networks
            TestNNXOR.tryTest(); // OK
            TestNNMnistDigits.tryTest(); // OK SLOW

            // Testing Deep 
            TestDQNNXOR.tryTest(); // OK Unstable learning
            TestDQNNMnistDigits.tryTest();  // OK SLOW Hyper sensitive to hyper parameters


            TestDQNCartPole.tryTest();
            TestQLearning.tryTest();
            TestGA.tryTest();
            */
            
            
        }
    }
}
