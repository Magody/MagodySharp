using System;
using System.Collections;
using System.Collections.Generic;

using System.Text.Json;

namespace types
{
    public class MatrixI : List<VectorI>
    {
        public static MatrixI parseJsonElement(JsonElement element){
            return JsonSerializer.Deserialize<MatrixI>(element.GetRawText());
        }
    }
    public class MatrixD : List<VectorD>
    {
        public static MatrixD parseJsonElement(JsonElement element){
            return JsonSerializer.Deserialize<MatrixD>(element.GetRawText());
        }

        public static MatrixD transposeVector(VectorD v){
            // vector to matrixD vertical (vector transpose)
            MatrixD matrix = new MatrixD();

            for (var i = 0; i < v.Count; i++)
            {
                matrix.Add(new VectorD{v[i]});
            }

            return matrix;
        }

        public static MatrixD clone(MatrixD m){

            if(m == null){
                return new MatrixD();
            }

            MatrixD matrixClone = new MatrixD();

            for (var i = 0; i < m.Count; i++)
            {
                matrixClone.Add(VectorD.clone(m[i]));
            }
            return matrixClone;
        }

        public static MatrixD slice(MatrixD source, int r_from, int r_to, int c_from, int c_to){
            MatrixD matrix = new MatrixD();

            for (var i = r_from; i <= r_to; i++)
            {
                matrix.Add(VectorD.slice(source[i], c_from, c_to));             
            }

            return matrix;
        }

        public static MatrixD filled(int n, int m, double value){
            MatrixD matrix = new MatrixD();

            for (var i = 0; i < n; i++)
            {
                matrix.Add(VectorD.filled(m,value));
            }

            return matrix;
        }

        public static MatrixD ones(int n, int m){
            MatrixD matrix = new MatrixD();

            double val = 1;
            for (var i = 0; i < n; i++)
            {
                matrix.Add(VectorD.filled(m,val));
            }

            return matrix;
        }

        public static MatrixD zeros(int n, int m){
            MatrixD matrix = new MatrixD();

            double val = 0;
            for (var i = 0; i < n; i++)
            {
                matrix.Add(VectorD.filled(m,val));
            }

            return matrix;
        }


        public MatrixD max(int dimension){
            MatrixD output = new MatrixD();

            VectorD values = new VectorD();
            VectorD indexes = new VectorD();

            if(dimension == 1){
                // generate vector with max value of each column

                for (var j = 0; j < this[0].Count; j++)
                {
                    double value = -1;
                    double index = -1;
                    for (var i = 0; i < this.Count; i++)
                    {
                        double value_new = this[i][j];
                        if(value_new > value){
                            value = value_new;
                            index = i;
                        }
                        
                    }
                    values.Add(value);
                    indexes.Add(index);
                }

                

            }
            else{
                throw new Exception("Not supported dimension in MatrixD max");
            }

            output.Add(values);
            output.Add(indexes);

            return output;
        }


        public MatrixD transpose(){
            MatrixD transposed = new MatrixD();
            int n = this.Count;
            if(n == 0) return transposed;
            int m = this[0].Count;

            for (var j = 0; j < m; j++)
            {
                VectorD v = new VectorD();
                for (var i = 0; i < n; i++)
                {
                    v.Add(this[i][j]);
                }
                transposed.Add(v);
            }
            return transposed;
        }

        public static MatrixD dot(MatrixD matrix1, MatrixD matrix2){
            
            MatrixD result = new MatrixD();
            int n1 = matrix1.Count;
            int n2 = matrix2.Count;

            if(n1 == 0 || n2 == 0) return result;

            int m1 = matrix1[0].Count;
            int m2 = matrix2[0].Count;

            // final dimension n1xm1

            for (int i = 0; i < n1; i++)
            {
                result.Add(VectorD.dot(matrix1[i], matrix2[i]));               
            }

            return result;
        }

        public static MatrixD dotDivide(MatrixD matrix1, MatrixD matrix2){
            
            MatrixD result = new MatrixD();
            int n1 = matrix1.Count;
            int n2 = matrix2.Count;

            if(n1 == 0 || n2 == 0) return result;

            int m1 = matrix1[0].Count;
            int m2 = matrix2[0].Count;

            bool repeat = false;

            if(n1 != n2){
                if(n2 == 1 && n1 > 1){
                    repeat = true;
                }
                else{
                    return result;
                }
            }

            // final dimension n1xm1

            for (int i = 0; i < n1; i++)
            {
                if(repeat){
                    result.Add(VectorD.dotDivide(matrix1[i], matrix2[0]));  
                }
                else{
                    result.Add(VectorD.dotDivide(matrix1[i], matrix2[i]));  
                }             
            }

            return result;
        }

        public static MatrixD sqrt(MatrixD matrix){

            MatrixD result = new MatrixD();

            int n=matrix.Count;

            for (var i = 0; i < n; i++)
            {
                result.Add(VectorD.sqrt(matrix[i]));
            }
            return result;
            
        }

        public static MatrixD exp(MatrixD matrix){

            MatrixD result = new MatrixD();

            int n=matrix.Count;

            for (var i = 0; i < n; i++)
            {
                result.Add(VectorD.exp(matrix[i]));
            }
            return result;
            
        }

        public static MatrixD tanh(MatrixD matrix){

            MatrixD result = new MatrixD();

            int n=matrix.Count;

            for (var i = 0; i < n; i++)
            {
                result.Add(VectorD.tanh(matrix[i]));
            }
            return result;
            
        }

        public static MatrixD log(MatrixD matrix){

            MatrixD result = new MatrixD();

            int n=matrix.Count;

            for (var i = 0; i < n; i++)
            {
                result.Add(VectorD.log(matrix[i]));
            }
            return result;
            
        }

        public MatrixD mean(int dimension){
            // todo: link with VectorD mean
            // mean(2) is a column vector containing the mean of each row.
            MatrixD mean_result = new MatrixD();
            int n = this.Count;
            int m = this[0].Count;

            if(dimension == 1){
                // reduce vertical, return row

                VectorD v = new VectorD();

                for (var j = 0; j < m; j++)
                {
                    double result = 0;
                    for (var i = 0; i < n; i++)
                    {
                        result += this[i][j];
                    }
                    v.Add((double)result/m);
                    
                }
                mean_result.Add(v);
                
            }
            else if(dimension == 2){
                // reduce horizontal, return col
                for (var i = 0; i < n; i++)
                {
                    double result = 0;
                    for (var j = 0; j < m; j++)
                    {
                        double matrix_value = this[i][j];
                        result += matrix_value;
                    }
                    mean_result.Add(new VectorD{(double)result/m});

                }
            }
            else{
                throw new Exception("Not supported");
            }

            return mean_result;
        }

        public MatrixD sum(int dimension){
            // todo: link to vector
            MatrixD result_sum = new MatrixD();
            int n = this.Count;
            int m = this[0].Count;

            if(dimension == 1){
                VectorD v = new VectorD();

                for (var j = 0; j < m; j++)
                {
                    double result = 0;
                    for (var i = 0; i < n; i++)
                    {
                        double matrix_value = this[i][j];
                        result += matrix_value;
                    }
                    v.Add(result);
                }
                result_sum.Add(v);
            }
            else if(dimension == 3){
                double sum = 0;
                foreach (var v in this)
                {
                    foreach (var value in v)
                    {
                        sum += value;
                    }
                }
                result_sum.Add(new VectorD{sum});
            }
            else{
                throw new Exception("Not supported");
            }

            return result_sum;
        }

        // compiler and others
        public static MatrixD operator *(MatrixD value, double scalar){
            
            MatrixD result = new MatrixD();
            int n = value.Count;

            if(n == 0) return result;

            int m = value[0].Count;

            double dynamic_scalar = scalar;

            for (int i = 0; i < n; i++)
            {
                VectorD v = value[i];
                result.Add(v * dynamic_scalar);                
            }

            return result;
        }

        public static MatrixD operator /(MatrixD value, double scalar){
            
            MatrixD result = new MatrixD();
            int n = value.Count;

            if(n == 0) return result;

            int m = value[0].Count;

            double dynamic_scalar = scalar;

            for (int i = 0; i < n; i++)
            {
                VectorD v = value[i];
                result.Add(v / dynamic_scalar);                
            }

            return result;
        }

        public static MatrixD operator +(MatrixD value, double scalar){
            
            MatrixD result = new MatrixD();
            int n = value.Count;

            if(n == 0) return result;

            int m = value[0].Count;

            double dynamic_scalar = scalar;

            for (int i = 0; i < n; i++)
            {
                VectorD v = value[i];
                result.Add(v + dynamic_scalar);                
            }

            return result;
        }

        public static MatrixD operator *(MatrixD matrix1, MatrixD matrix2){
            
            MatrixD result = new MatrixD();
            int n1 = matrix1.Count;
            int n2 = matrix2.Count;

            if(n1 == 0 || n2 == 0) return result;

            int m1 = matrix1[0].Count;
            int m2 = matrix2[0].Count;

            // final dimension n1xm2

            for (int i = 0; i < n1; i++)
            {
                VectorD v = new VectorD();

                for (var j = 0; j < m2; j++)
                {
                    double temp = 0;

                    for (int k = 0; k < m1; k++)
                    {
                        double matrix1_value = matrix1[i][k];
                        double matrix2_value = matrix2[k][j];

                        temp += matrix1_value * matrix2_value;
                    }
                    v.Add(temp);
                }
                result.Add(v);                
            }

            return result;
        }


        public static MatrixD operator +(MatrixD matrix1, MatrixD matrix2){
            
            MatrixD result = new MatrixD();
            int n1 = matrix1.Count;
            int n2 = matrix2.Count;

            if(!(n1 == n2) || n1 == 0) return result;

            int m1 = matrix1[0].Count;
            int m2 = matrix2[0].Count;

            
            if(m1 == 0) return result;

            bool repeat = false;

            if(m2 != m1){
                if(m2 == 1 && m1 > 1){
                    repeat = true;
                }
                else{
                    return result;
                }
            }

            // Console.WriteLine($"repeat{repeat}");


            for (int i = 0; i < n1; i++)
            {
                VectorD v = new VectorD();

                for (var j = 0; j < m1; j++)
                {
                    double matrix1_value = matrix1[i][j];
                    
                    double matrix2_value;

                    if(repeat){
                        matrix2_value = matrix2[i][0];
                    }
                    else{
                        matrix2_value = matrix2[i][j];
                    }

                    v.Add(matrix1_value + matrix2_value);
                    
                }
                result.Add(v);                
            }

            return result;
        }

        public static MatrixD operator -(MatrixD matrix1, MatrixD matrix2){
            
            MatrixD result = new MatrixD();
            int n1 = matrix1.Count;
            int n2 = matrix2.Count;

            if(!(n1 == n2) || n1 == 0) return result;

            int m1 = matrix1[0].Count;
            int m2 = matrix2[0].Count;

            
            if(!(m1 == m2) || m1 == 0) return result;


            for (int i = 0; i < n1; i++)
            {
                VectorD v = new VectorD();

                for (var j = 0; j < m1; j++)
                {
                    double matrix1_value = matrix1[i][j];
                    double matrix2_value = matrix2[i][j];
                    v.Add(matrix1_value - matrix2_value);
                    
                }
                result.Add(v);                
            }

            return result;
        }

        public override string ToString(){
            int length = this.Count;

            string output = "";
            for (var i = 0; i < length; i++)
            {
                var item = this[i];
                output += item.ToString();
                if(i< length-1){
                    output += ",";
                }
            }
            return $"[{output}]";
        }
    }
}
