using System;
using System.Collections;
using System.Collections.Generic;

using System.Text.Json;

namespace types
{
    
    public class VectorI : List<int>
    {

        public static VectorI parseJsonElement(JsonElement element){
            return JsonSerializer.Deserialize<VectorI>(element.GetRawText());
        }

        public static VectorI clone(VectorI v){

            VectorI cloned = VectorI.zeros(v.Count);
            for (var i = 0; i < v.Count; i++)
            {
                cloned[i] = v[i];
            }
            return cloned;
        }
        public static VectorI filled(int n, int value){
            VectorI v = new VectorI();

            for (var i = 0; i < n; i++)
            {
                v.Add(value);
            }

            return v;
        }

        public static VectorI ones(int n){
            // todo: concatenate this with Matrix.ones
            return VectorI.filled(n,1);
        }

        public static VectorI zeros(int n){
            // todo: concatenate this with Matrix.zeros
            return VectorI.filled(n,0);
        }

        public override string ToString(){
            int length = this.Count;

            string output = "";
            for (var i = 0; i < length; i++)
            {
                var item = this[i];
                output += item;
                if(i< length-1){
                    output += ",";
                }
            }
            return $"[{output}]";
        }

    }
    public class VectorD : List<double>
    {
        public float[] parseToFloatArray(){
            float[] array = new float[this.Count];

            for (int i = 0; i < this.Count; i++)
            {
                array[i] = (float)this[i];
            }
            return array;
        }
        public static VectorD parseFloatArray(float[] array){
            VectorD vectorD = new VectorD();

            foreach (var item in array)
            {
                vectorD.Add((double)item);
            }
            return vectorD;
        }

        public static VectorD parseJsonElement(JsonElement element){
            return JsonSerializer.Deserialize<VectorD>(element.GetRawText());
        }

        public static VectorD clone(VectorD v){

            if(v == null){
                return new VectorD();
            }

            VectorD cloned = VectorD.zeros(v.Count);
            for (var i = 0; i < v.Count; i++)
            {
                cloned[i] = v[i];
            }
            return cloned;
        }

        public static VectorD slice(VectorD source, int from, int to){
            // VectorD res = (VectorD)source.GetRange(from, to - from + 1);
            VectorD res = new VectorD();
            for (var i = from; i <= to; i++)
            {
                res.Add(source[i]);
            }
            return res;
        }

        public static VectorD filled(int n, double value){
            VectorD v = new VectorD();

            for (var i = 0; i < n; i++)
            {
                v.Add(value);
            }

            return v;
        }

        public static VectorD ones(int n){
            // todo: concatenate this with Matrix.ones
            return VectorD.filled(n,1);
        }

        public static VectorD zeros(int n){
            // todo: concatenate this with Matrix.zeros
            return VectorD.filled(n,0);
        }

        public static Dictionary<string,double> min(VectorD v){

            Dictionary<string,double> result = new Dictionary<string, double>();

            double min_index = -1;
            double min_value = double.MaxValue;

            for (var i = 0; i < v.Count; i++)
            {
                double value = v[i];
                if(value < min_value){
                    min_value = value;
                    min_index = i;
                }
            }

            result["value"] = min_value;
            result["index"] = min_index;

            return result;
        }

        public static Dictionary<string,double> max(VectorD v){

            Dictionary<string,double> result = new Dictionary<string, double>();

            double max_index = -1;
            double max_value = double.MinValue;

            for (var i = 0; i < v.Count; i++)
            {
                double value = v[i];
                if(value > max_value){
                    max_value = value;
                    max_index = i;
                }
            }

            result["value"] = max_value;
            result["index"] = max_index;

            return result;
        }

        
        public static VectorD sqrt(VectorD v){

            VectorD result = new VectorD();

            for (var i = 0; i < v.Count; i++)
            {
                double value = v[i];
                result.Add(Math.Sqrt(value));
            }

            return result;
        }

        public static VectorD exp(VectorD v){

            VectorD result = new VectorD();

            for (var i = 0; i < v.Count; i++)
            {
                double value = v[i];
                result.Add(Math.Exp(value));
            }

            return result;
        }

        public static VectorD tanh(VectorD v){

            VectorD result = new VectorD();

            for (var i = 0; i < v.Count; i++)
            {
                double value = v[i];
                result.Add(Math.Tanh(value));
            }

            return result;
        }

        public static VectorD cosh(VectorD v){

            VectorD result = new VectorD();

            for (var i = 0; i < v.Count; i++)
            {
                result.Add(Math.Cosh(v[i]));
            }

            return result;
        }

        public static VectorD log(VectorD v){

            VectorD result = new VectorD();

            for (var i = 0; i < v.Count; i++)
            {
                result.Add(Math.Log(v[i]));
            }

            return result;
        }

        public static double sum(VectorD v){

            double result = 0;

            for (var i = 0; i < v.Count; i++)
            {
                result += v[i];
            }

            return result;
        }


        public static VectorD dot(VectorD v1, VectorD v2){
            
            VectorD result = new VectorD();
            int n1 = v1.Count;
            int n2 = v2.Count;

            if(n1 == 0 || n2 == 0) return result;

            for (int i = 0; i < n1; i++)
            {
                double value1 = v1[i];
                double value2 = v2[i];
                result.Add(value1 * value2);                
            }

            return result;
        }

        public static VectorD dotDivide(VectorD v1, VectorD v2){
            
            VectorD result = new VectorD();
            int n1 = v1.Count;
            int n2 = v2.Count;

            if(n1 == 0 || n2 == 0) return result;

            for (int i = 0; i < n1; i++)
            {
                double value1 = v1[i];
                double value2 = v2[i];
                result.Add((double)value1 / value2);                
            }

            return result;
        }

        public static double mean(VectorD v){
            double mean_result = 0;
            int n = v.Count;

            for (var i = 0; i < n; i++)
            {
                mean_result += v[i];
            }

            return mean_result/n;
        }


        public static VectorD operator +(VectorD v1, VectorD v2){
            
            VectorD result = new VectorD();
            int length = v1.Count;

            if(length == 0) return result;

            for (int i = 0; i < length; i++)
            {
                double dynamic1 = v1[i];
                double dynamic2 = v2[i];
                result.Add(dynamic1 + dynamic2);
            }

            return result;
        }

        public static VectorD operator -(VectorD v1, VectorD v2){
            
            VectorD result = new VectorD();
            int length = v1.Count;

            if(length == 0) return result;

            for (int i = 0; i < length; i++)
            {
                double dynamic1 = v1[i];
                double dynamic2 = v2[i];
                result.Add(dynamic1 - dynamic2);
            }

            return result;
        }

        public static VectorD operator *(VectorD v1, double scalar){
            
            VectorD result = new VectorD();
            int length = v1.Count;

            if(length == 0) return result;

            double dynamic_scalar = scalar;

            for (int i = 0; i < length; i++)
            {
                double dynamic1 = v1[i];
                result.Add(dynamic1 * dynamic_scalar);
            }

            return result;
        }

        public static VectorD operator /(VectorD v1, double scalar){
            
            VectorD result = new VectorD();
            int length = v1.Count;

            if(length == 0) return result;

            double dynamic_scalar = scalar;

            for (int i = 0; i < length; i++)
            {
                double dynamic1 = v1[i];
                result.Add((double)dynamic1 / dynamic_scalar);
            }

            return result;
        }

        public static VectorD operator +(VectorD v1, double scalar){
            
            VectorD result = new VectorD();
            int length = v1.Count;

            if(length == 0) return result;

            double dynamic_scalar = scalar;

            for (int i = 0; i < length; i++)
            {
                double dynamic1 = v1[i];
                result.Add(dynamic1 + dynamic_scalar);
            }

            return result;
        }

        public override string ToString(){
            int length = this.Count;

            string output = "";
            for (var i = 0; i < length; i++)
            {
                var item = this[i];
                output += item;
                if(i< length-1){
                    output += ",";
                }
            }
            return $"[{output}]";
        }
        
    }

    
}
