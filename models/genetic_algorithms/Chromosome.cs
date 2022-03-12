using System;
using System.Collections;
using System.Collections.Generic;

using types;
using custom_lib;


namespace genetic_algoritms
{
    public class Chromosome
    {
        public VectorD gens {get; set;}
        private Random randomGenerator;


        public Chromosome(VectorD gens){
            this.gens = gens;
            int len_gens_set = this.gens.Count;
            this.randomGenerator = new Random();
        }

        public Chromosome(MatrixD gens_set){
            int len_gens_set = gens_set.Count;
            this.gens = new VectorD();
            this.randomGenerator = new Random();

            for (var i = 0; i < len_gens_set; i++)
            {
                this.gens.Add(Chromosome.chooseRandomSample(this.randomGenerator, gens_set[i], 1)[0]);
            }

        }

        public Chromosome clone(){
            Chromosome cloned = new Chromosome(VectorD.clone(this.gens));
            return cloned;
        }


        public static VectorD chooseRandomSample(Random random, VectorD gen_set, int size){

           VectorD sample = VectorD.zeros(size);
           int len_set = gen_set.Count;

           for (var i = 0; i < size; i++)
           {
               int index = random.Next(0, len_set);
               sample[i] = gen_set[index];           
           }
           return sample;
        }

        public override string ToString(){
            string output = "";
            output += $"gens {gens}";
            return output;
        }
        
    }
}
