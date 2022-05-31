
using System.Threading;
using types;

namespace reinforcement_learning
{

    public abstract class Agent
    {
        public delegate void CallbackOnEpisodeBegin(int episode, EnvironmentRL env);
        public delegate float[] CallbackOnCollectObservations(int episode, EnvironmentRL env);
        public delegate void CallbackOnActionPerformed(int episode, EnvironmentRL env, float[] actions_values);
        public delegate void CallbackOnEpisodeEnd(int episode, EnvironmentRL env);

        public CallbackOnEpisodeBegin onEpisodeBegin {get;set;}
        public CallbackOnCollectObservations onCollectObservations {get;set;}
        public CallbackOnActionPerformed onActionPerformed {get;set;}
        public CallbackOnEpisodeEnd onEpisodeEnd {get;set;}
        public int decision_period_milliseconds {get;set;}
        public int maximum_steps {get;set;}
        public bool is_episode_running {get;set;}

        public Agent(
            CallbackOnEpisodeBegin onEpisodeBegin,
            CallbackOnCollectObservations onCollectObservations,
            CallbackOnActionPerformed onActionPerformed,
            CallbackOnEpisodeEnd onEpisodeEnd,
            int decision_period_milliseconds = 1000, // milliseconds
            int maximum_steps = 10
        ){
            this.onEpisodeBegin = onEpisodeBegin;
            this.onCollectObservations = onCollectObservations;
            this.onActionPerformed = onActionPerformed;
            this.onEpisodeEnd = onEpisodeEnd;
            this.decision_period_milliseconds = decision_period_milliseconds;
            this.maximum_steps = maximum_steps;
            this.is_episode_running = false;
        }        

            
    }
}
