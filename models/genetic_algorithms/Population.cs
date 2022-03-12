using System;
using System.Collections;
using System.Collections.Generic;

using types;
using custom_lib;


namespace genetic_algoritms
{

    public enum MethodSelection
    {
        elitist
    }

    public class Population
    {
        public double threshold_fitness {get; set;}

        public List<Chromosome> chromosomes {get; set;}
        public MatrixD gens_set {get; set;}

        public int max_population = 0;
        public float mutation_rate = 0.9f;
        public delegate double FitnessFunction(Chromosome chromosome, Dictionary<string,string> context);

        public FitnessFunction fitnessFunction {get;set;}

        public Dictionary<int,VectorD> changes {get;set;}
        private Random randomGenerator;

        public Population(int max_population, MatrixD gens_set, float mutation_rate, FitnessFunction fitnessFunction, double threshold_fitness=-1){

            this.max_population = max_population;
            this.gens_set = gens_set;
            this.mutation_rate = mutation_rate;
            this.fitnessFunction = fitnessFunction;
            this.threshold_fitness = threshold_fitness;
            this.randomGenerator = new Random();
            this.chromosomes = new List<Chromosome>();
            changes = new Dictionary<int, VectorD>();
            int len_gens = this.gens_set.Count;
            for (var i = 0; i < len_gens; i++)
            {
                this.changes[i] = new VectorD();
            }

            this.generateInitialPopulation();

        }

        public void generateInitialPopulation(){
            this.chromosomes.Clear();

            for (var i = 0; i < this.max_population; i++)
            {
                this.chromosomes.Add(new Chromosome(this.gens_set));
            }
        }

        public VectorD evaluateFitness(Dictionary<string,string> context){

            VectorD fitness = new VectorD();

            for (var i = 0; i < this.max_population; i++)
            {
                Chromosome chromosome = this.chromosomes[i];
                fitness.Add(this.fitnessFunction(chromosome, context));
            }

            return fitness;


        }

        public List<Chromosome> mutate(List<Chromosome> offspring_crossover, float reductor_factor){
            
            List<Chromosome> offspring_mutated = new List<Chromosome>();
            // Mutation changes a single gene in each offspring randomly.
            int len_individuals = offspring_crossover.Count;
            int len_gens = offspring_crossover[0].gens.Count;

            int[] poles = new int[2]{-1,1};
            int amount_sample = this.gens_set[0].Count;

            for (var i = 0; i < len_individuals; i++)
            {

                offspring_mutated.Add(offspring_crossover[i].clone());

                double dice = this.randomGenerator.NextDouble();
                if(dice <= this.mutation_rate){
                    // it's time to mutate some gen/gens
                    int index_gen = this.randomGenerator.Next(0, len_gens);

                    double minimum_valid_number = this.gens_set[index_gen][0];
                    double maximum_valid_number = this.gens_set[index_gen][amount_sample-1];
                    
                    double change = (double)((maximum_valid_number+minimum_valid_number)/2) * reductor_factor; 
                    int direction = poles[this.randomGenerator.Next(0,2)];
                    if(change == 0){
                        change = ((double)(Math.Abs(maximum_valid_number)+Math.Abs(minimum_valid_number))/((double)maximum_valid_number/50)) * reductor_factor;
                    }

                    double change_direction = (change * (double)direction);

                    this.changes[index_gen].Add(change_direction);

                    double change_final = offspring_mutated[i].gens[index_gen] + change_direction;

                    // Math.Clamp no exist in net standard
                    if(change_final > maximum_valid_number){
                        change_final = maximum_valid_number;
                    }
                    else if(change_final < minimum_valid_number){
                        change_final = minimum_valid_number;
                    }

                    offspring_mutated[i].gens[index_gen] = change_final; // Math.Clamp(change_final, minimum_valid_number, maximum_valid_number);

                }

                
                
            }



            return offspring_mutated;
        }

        public bool hasReachedTheTop(VectorD fitness){
            if(this.threshold_fitness == -1) return false;
            double fitness_mean = VectorD.mean(fitness);
            return fitness_mean >= this.threshold_fitness;
        }


        public static VectorI selectBestIndividuals(MethodSelection method, VectorD fitness, int num_best_individuals){
            // elitist method only select the two best parents 
            
            VectorI best_individuals_index = VectorI.zeros(num_best_individuals);

            switch (method)
            {
                case MethodSelection.elitist:
                    // very low numbers
                    VectorD best_individuals_values = VectorD.filled(num_best_individuals, double.MinValue);
                    
                    for (var i = 0; i < fitness.Count; i++)
                    {
                        double fitness_individual = fitness[i];

                        Dictionary<string,double> firstMin = VectorD.min(best_individuals_values);
                        double first_min_value = firstMin["value"];
                        int first_min_index = (int)firstMin["index"];

                        if(fitness_individual > first_min_value){
                            best_individuals_values[first_min_index] = fitness_individual;
                            best_individuals_index[first_min_index] = i;
                        }
                        
                    }
                    break;
                default:
                    throw new Exception("Method not supported, see Enum");
            }

            return best_individuals_index;
            

        }

        public static List<Chromosome> crossover(List<Chromosome> parents, int num_individuals){
            // IT CAN BE IMPROVED. Only making combinations of 2 parents
            int len_parents = parents.Count;
            int len_gens = parents[0].gens.Count;
            
            List<Chromosome> offspring = new List<Chromosome>();
            
            // The point at which crossover takes place between two parents. 
            // Usually, it is at the center. There are a lot of methods
            // better than this one.
            
            int crossover_point = (int)Math.Floor((double)len_gens/2);

            // its a secuencial crossover, which is bad due to prob. duplication

            for (var i = 0; i < num_individuals; i++)
            {
                int parent1_idx = (i % len_parents);
                int parent2_idx = ((i+1) % len_parents);
                // The new offspring will have its first half of its genes taken from
                offspring.Add(parents[parent1_idx].clone());  // IMPORTANT, NO PASS BY REFERENCE
                // The new offspring will have its second half of its genes taken from
                for (var j = crossover_point; j < len_gens; j++)
                {
                    offspring[i].gens[j] = parents[parent2_idx].gens[j];
                }
                
            }

            return offspring;
        }

         
            

    }
}
