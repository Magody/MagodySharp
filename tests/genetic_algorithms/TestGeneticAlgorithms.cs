using System;
using System.Collections.Generic;

using layers;
using types;
using custom_lib;
using genetic_algoritms;


namespace lightmlsharp
{
    class TestGA
    {
        
        public static void tryTest()
        {
            
            int verbose_level = 2;

            int generations = 100;  // its just a stop step in case the program cant find solution
            int max_population = 30;  // while more high, more posibilities to find a good combination faster, but more processing
            int num_parents_to_select = 4;  // is better tu select very little, no more than 10// of max population
            float mutation_rate = 0.9f;
            int amount_examples = 100;

            
            // examples of parameters
            MatrixD gens_set = new MatrixD();
            gens_set.Add(MathUtil.linspace(-100,100,amount_examples)); // allowed range for x

            int len_gens_set = gens_set.Count;

            Population.FitnessFunction fitnessFunction = (Chromosome chromosome, Dictionary<string,string> context) => {
                // optimice Max: y = (x^2)*-1 + 3 + (cos(x)*4)^2 (EXPECTED: x=0,y=19)
                double fitness_score = 0;

                double x = chromosome.gens[0];
                fitness_score = -Math.Pow(x, 2) + Math.Pow(Math.Cos(x) * 4, 2) + 3;

                return fitness_score;
            };

            Dictionary<string,string> context = new Dictionary<string, string>();
            context["verbose"] = $"{verbose_level}";

            Population population = new Population(max_population, gens_set, mutation_rate, fitnessFunction);
            

            VectorD history_fitness_mean = new VectorD();
            DateTime start = DateTime.UtcNow;

            VectorD fitness = new VectorD();

            for (var generation = 0; generation < generations; generation++)
            {
                Console.WriteLine($"Generation {generation+1} of {generations}");
                fitness = population.evaluateFitness(context);
                
                // being elitist means that the parents are the best 2 (maybe a little more)
                
                VectorI best_individuals_index = Population.selectBestIndividuals(MethodSelection.elitist, fitness, num_parents_to_select);
                
                List<Chromosome> parents_selected = new List<Chromosome>();
                for (var i = 0; i < best_individuals_index.Count; i++)
                {
                    parents_selected.Add(population.chromosomes[best_individuals_index[i]]);
                }

                
                
                ////////////////////////////RESULTS ACTUAL GENERATION//////////////////////////////////
                
                Console.WriteLine($"Best results for generation {generation+1}");
                string output = "";
                for (var i = 0; i < best_individuals_index.Count; i++)
                {
                    output += fitness[best_individuals_index[i]] + " -> ";
                    
                }
                Console.WriteLine(output);

                history_fitness_mean.Add(VectorD.mean(fitness));
                ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                
                
                if(population.hasReachedTheTop(fitness)){
                    Console.WriteLine("EARLY END");
                    break;
                }
                
                if(generation < generations){
                    List<Chromosome> offspring_crossover = Population.crossover(parents_selected, max_population-num_parents_to_select);
                
                    // the change for the alele of gen, higher at the begining (exploration), 
                    // lower at the end (lower exploration, high explotation)
                    float reductor_factor = 0;

                    if(generation < generations * 0.6){
                        reductor_factor = (float)1/(1+generation*2);
                    }
                    else if(generation < generations * 0.8){
                        reductor_factor = (float)1/(1+generation*6);
                    }
                    else{
                        // more fine search
                        reductor_factor = (float)1/((float)Math.Pow(1+generation,2)*1.2f);
                    }
                    
                    

                    List<Chromosome> offspring_mutated = population.mutate(offspring_crossover, reductor_factor);  // Creating the new population based on the parents and offspring.

                    for (var i = 0; i < num_parents_to_select; i++)
                    {
                        population.chromosomes[i] = parents_selected[i];
                    }
                    for (var i = num_parents_to_select; i < population.chromosomes.Count; i++)
                    {
                        population.chromosomes[i] = offspring_mutated[i-num_parents_to_select];
                    }
                
                }
            }     

            DateTime end = DateTime.UtcNow;
            TimeSpan timeDiff = end - start;
            int time_final = Convert.ToInt32(timeDiff.TotalMilliseconds);
            Console.WriteLine($"Elapsed time: {time_final} ms");

            int index_best_chromosome = Population.selectBestIndividuals(MethodSelection.elitist, fitness, 1)[0];
            
            Console.WriteLine($"Best Chromosome {population.chromosomes[index_best_chromosome]} ");
            Console.WriteLine($"Best Fitness {fitness[index_best_chromosome]}");


            Console.WriteLine(history_fitness_mean);

            VectorD uniques = new VectorD();
            double last = -1;
            for (var i = 0; i < population.changes[0].Count; i++)
            {
                double value = Math.Abs(population.changes[0][i]);
                if(value != last && !uniques.Contains(value)){
                    uniques.Add(value);
                }
                last = value;
                
            }
            Console.WriteLine(uniques);

            
        }
    }
}
