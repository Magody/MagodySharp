using System;
using System.Threading;
using System.Collections.Generic;
using System.Collections;
using types;

namespace reinforcement_learning
{

    public class AgentDQNN : Agent
    {
        public VectorD update_costs {get;set;}
        private QNeuralNetwork qNeuralNetwork;

        public AgentDQNN(QNeuralNetwork qNeuralNetwork, CallbackOnEpisodeBegin onEpisodeBegin, CallbackOnCollectObservations onCollectObservations, CallbackOnActionPerformed onActionPerformed, CallbackOnEpisodeEnd onEpisodeEnd, int decision_period = 1000, int maximum_steps = 10) : base(onEpisodeBegin, onCollectObservations, onActionPerformed, onEpisodeEnd, decision_period, maximum_steps)
        {
            this.qNeuralNetwork = qNeuralNetwork;
        }


        public void runEpisodesAsync(EnvironmentRL env, int episodes, QLearning.FunctionReward functionReward){
            
            this.update_costs = new VectorD();

            for (int i = 0; i < episodes; i++)
            {
                int episode = i+1;
                this.is_episode_running = true;

                env.reset(new Hashtable{});
                this.qNeuralNetwork.updateEpsilonDecay(episode);

                int step_discrete = 0;

                VectorD update_costs_episode = new VectorD(); 
                this.onEpisodeBegin(episode, env);
                while(env.timestep < this.maximum_steps && is_episode_running){
                    // is_episode_running can be changed from outside! with endEpisode
                    float[] observations = this.onCollectObservations(episode, env);
                    VectorD state = VectorD.parseFloatArray(observations);

                    QSelection qSelection = qNeuralNetwork.selectAction(state, false);
                    this.onActionPerformed(episode, env, qSelection.q.parseToFloatArray());
                    int action = qSelection.action_index;
                    ReplayTuple newReplayTuple = functionReward(qNeuralNetwork, state, action, env);
                    env.reward += newReplayTuple.reward;

                    // STORING THIS NEW TRANSITION INTO THE MEMORY
                    if(env.is_train_only){
                        qNeuralNetwork.saveReplayTuple(newReplayTuple);

                        // train in intervals to optimice
                        if(step_discrete % qNeuralNetwork.qLearningConfig.interval_for_learning == 0){
                            ExperienceReplayHistoryLearning history_learning = qNeuralNetwork.learnFromExperienceReplay(episode, env.verbose_level-1);
                            if(history_learning.learned){
                                update_costs_episode.Add(history_learning.mean_cost);
                            }
                        }

                        if(step_discrete % qNeuralNetwork.qLearningConfig.interval_for_update_models == 0){
                            qNeuralNetwork.updateQNeuralNetworkTarget();
                        }
                    }

                    
                    // string summary = $"s:{newReplayTuple.state} -> a:{newReplayTuple.action} -> r:{newReplayTuple.reward} -> s':{newReplayTuple.new_state} -> End:{newReplayTuple.is_terminal}";
                    // Console.WriteLine(summary);

                    is_episode_running = !newReplayTuple.is_terminal;
                    if(env.auto_increase_timestep){
                        env.timestep += 1;
                    }
                    // step_discrete always is increased by one
                    step_discrete+=1;
                    
                    if(this.decision_period_milliseconds > 0){
                        Thread.Sleep(this.decision_period_milliseconds);
                    }

                    if(!is_episode_running){
                        // Console.WriteLine($"END EPISODE PRE {qSelection.q}");
                    }
                }
                // Console.WriteLine(update_costs_episode);
                double cost = -1;
                if(update_costs_episode.Count > 0){
                    cost = VectorD.mean(update_costs_episode);
                }
                this.update_costs.Add(cost);
                env.cost=(float)cost;
                this.onEpisodeEnd(episode, env);

                this.is_episode_running = false;
                
            }

                       



        }
    }


}