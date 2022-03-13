using System;
using System.Collections;
using System.Collections.Generic;

using types;


namespace custom_lib
{
    public class Utils
    {
        public static string getJsonAttr(string name, string value, bool is_end=false){
            return $"\t\"{name}\": {value}" + ((is_end)? "\n":",\n");
        }

        public static VectorD oneHotEncodingVector(int position, int number_clases){
            VectorD vector = VectorD.filled(number_clases, 0);
            vector[position] = 1;
            return vector;
        }



    }
}
