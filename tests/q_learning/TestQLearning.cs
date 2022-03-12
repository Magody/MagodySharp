using System;
using System.Collections.Generic;
using System.Collections;

using types;
using custom_lib;
using reinforcement_learning;
using neural_network;
using layers;
using loss;

using test;


using System.Text.Json;




namespace magodysharp
{
    

    class TestQLearning
    {

        public static void qLearningTablePathFinding(){
            /*
            int num_nodes = 9;
            int edges = 10;

            int[][] environment_graph_literal = new int[edges][];
            environment_graph_literal[0] = new int[2]{0,1};
            environment_graph_literal[1] = new int[2]{1,5};
            environment_graph_literal[2] = new int[2]{5,6};
            environment_graph_literal[3] = new int[2]{5,4};
            environment_graph_literal[4] = new int[2]{1,2};
            environment_graph_literal[5] = new int[2]{2,3};
            environment_graph_literal[6] = new int[2]{2,7};
            environment_graph_literal[7] = new int[2]{8,0};
            environment_graph_literal[8] = new int[2]{8,2};
            environment_graph_literal[9] = new int[2]{2,5};

            int goal = 6;  // X node to 6
            int[,] reward_matrix = new int[num_nodes,num_nodes];
            for (var i = 0; i < num_nodes; i++)
            {
                for (var j = 0; j < num_nodes; j++)
                {
                    reward_matrix[i,j] = -1;
                }
            }
            
            int[,] adjacency_matrix = new int[num_nodes,num_nodes];

            for (var i = 0; i < edges; i++)
            {
                int from = environment_graph_literal[i][0];
                int to = environment_graph_literal[i][1];
                adjacency_matrix[from, to] = 1;
                adjacency_matrix[to, from] = 1;
                
                if(to == goal)
                    reward_matrix[from, to] = 10;
                else
                    reward_matrix[from, to] = 0;
                
                if(from == goal)
                    reward_matrix[to, from] = 10;
                else
                    reward_matrix[to, from] = 0;
            }

            reward_matrix[goal, goal] = 10;

            
            int verbose_level = 10;
            Hashtable context = new Hashtable();
            context["reward_matrix"] = reward_matrix;
            context["num_actions"] = num_nodes;
            context["terminal"] = goal;

            // Init hyper parameters
            float gamma = 0.9f;
            float epsilon = 1f;
            int total_episodes = 10000;
            // no experience replay
            GameReplayStrategy gameReplayStrategy = GameReplayStrategy.linear;
            int experience_replay_reserved_space = 0;

            QLearningConfig qLearningConfig = new QLearningConfig(gamma, epsilon, total_episodes, gameReplayStrategy, experience_replay_reserved_space);
            QLearningTable qLearning = new QLearningTable(qLearningConfig);

            QLearningTable.FunctionReward functionReward = (VectorD state, int action_selected, Hashtable context) => {
                    
                    Hashtable outputReward = new Hashtable();

                    int terminal = (int)context["terminal"];
                    int[,] reward_matrix = (int[,])context["reward_matrix"];
                    double reward = reward_matrix[(int)state[0], action_selected];
                    
                    VectorD new_state;
                    if(reward == -1)
                        new_state = state;
                    else
                        new_state = new VectorD{action_selected};
                    
                    bool finish = action_selected == terminal && reward >= 0;

                    outputReward["reward"] = reward;
                    outputReward["new_state"] = new_state;
                    outputReward["finish"] = finish;

                    return outputReward;
                    
                };


            // Train
            Hashtable history_episodes = qLearning.runEpisodes(
                functionReward, false, context, verbose_level-1
            );

            // get table
            MatrixD Q_table = (MatrixD)history_episodes["Q_table"];

          
            // test

            // 8 -> 6, expected: 8->2->5->6

            int state = 8;
            List<int> best_path = new List<int>{state};

            Console.WriteLine($"Best path for begin in {state}");

            for (var i = 0; i < num_nodes; i++)
            {
                for (var j = 0; j < num_nodes; j++)
                {
                    Console.Write($"{reward_matrix[i,j]} ");
                }
                Console.WriteLine();
            }
            
            Console.WriteLine(Q_table);

            for (var step = 1; step <= 5; step++)
            {
                Hashtable q_a = QLearning.selectActionQEpsilonGreedy(Q_table[state],qLearning.epsilon, num_nodes, true);
                state = (int)q_a["action_index"];
                best_path.Add(state);
                Console.WriteLine($"-> {state}");
            }
            */
        }
        
        public static void thompsonSamplingMaximizeRevenues(){
            int N = 10000;
            int d = 9;
            
            // Creating the simulation: This data should be collected

            Random randomGenerator = new Random();
            double[] conversion_rates = new double[]{0.05,0.13,0.09,0.16,0.11,0.04,0.20,0.08,0.01};
            
            double[][] X = new double[N][];
            for (var i = 0; i < N; i++)
            {
                X[i] = new double[d];
                for (var j = 0; j < d; j++)
                {
                    if(randomGenerator.NextDouble() <= conversion_rates[j]){
                        X[i][j] = 1;
                    }
                    else{
                        X[i][j] = 0;
                    }
                    
                }
                
            }
            
            ThompsonSampling thompsonSampling = new ThompsonSampling(X);

            Hashtable outputThompson = thompsonSampling.selectBest();
            VectorD histogram = (VectorD)outputThompson["histogram"];
            
            Console.WriteLine($"Original distribution [0.05,0.13,0.09,0.16,0.11,0.04,0.20,0.08,0.01]");
            Console.WriteLine($"Estimated distribution {histogram}");
            Console.WriteLine(outputThompson["return_absolute"]);
            Console.WriteLine(outputThompson["return_relative"]);
            Console.WriteLine(outputThompson["best_strategy"]);
        }
        
        public static void tryTest()
        {
            // Console.WriteLine(Bandit.kArmedBanditDemoSimulation());
            // Console.WriteLine(TestQLearning.qLearningTablePathFinding());
            // thompsonSamplingMaximizeRevenues();


        }
    }

    
}

