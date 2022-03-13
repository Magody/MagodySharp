using System;
using System.Collections;
using System.Collections.Generic;

using types;


namespace custom_lib
{
    public class MathUtil
    {
        private static Random randomGenerator = new Random();

        public static VectorD linspace(int begin, int end, int size){
            VectorD v = new VectorD();
            double diff = (float)(end - begin)/(size-1);

            v.Add(begin);

            for (var i = 0; i < size-2; i++)
            {
                v.Add(Math.Round(v[i]+diff, 2));
            }
            
            v.Add(end);

            return v;

        }

        public static float getRandomUniform(float low=0,float high=1){
            
            float random_sample = (float)MathUtil.randomGenerator.NextDouble(); //uniform(0,1]
            
            return low + (high-low) * random_sample;
        }

        public static double getRandomNormal(float mean=0.0f, float std=1.0f){
            
            // todo: repair random, parametize random
            float u1 = 1f-(float)MathUtil.randomGenerator.NextDouble(); //uniform(0,1] rand doubles
            float u2 = 1f-(float)MathUtil.randomGenerator.NextDouble();

            double rand_normal_std = Math.Sqrt(-2f * Math.Log(u1)) * Math.Sin(2f * Math.PI * u2); //random normal(0,1)
            double rand_normal = mean + std * rand_normal_std; //rand normal(mean,stdDev^2)
        
            return rand_normal;
        }

        public static VectorD getVectorRandomNormal(int length, float mean=0.0f, float std=1.0f){

            VectorD v = new VectorD();
            for (var i = 0; i < length; i++)
            {
                v.Add(MathUtil.getRandomNormal(mean,std));
            }

            return v;
        }

        public static MatrixD getMatrixRandomNormal(int n, int m, float mean=0.0f, float std=1.0f){
            
            MatrixD matrix = new MatrixD();

            for (var i = 0; i < n; i++)
            {
                VectorD v = new VectorD();
                for (var j = 0; j < m; j++)
                {
                    v.Add(MathUtil.getRandomNormal(mean,std));
                }
                matrix.Add(v);
                
            }

            return matrix;
        }




       

    }
}
