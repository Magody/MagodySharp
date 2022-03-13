using System;
using System.Collections;
using System.Collections.Generic;

using types;
using custom_lib;


namespace reinforcement_learning
{
    public enum TypeExecution
    {
        train, validation, test
    }
    public abstract class EnvironmentRL
    {
        public string name {get;set;}
        
        public TypeExecution typeExecution = TypeExecution.train;
        public int verbose_level = 1;

        public float cost = -1;
        public bool is_validation_or_test {get;set;}
        public bool is_train_only {get;set;}
        public bool is_validation_only {get;set;}
        public bool is_test_only {get;set;}
        public int patience_max {get;set;}
        public int patience {get;set;}

        public float reward_best_episode {get;set;}
        public float reward {get;set;}
        public bool game_over {get;set;}
        public float timestep {get;set;}

        public bool auto_increase_timestep {get;set;}

        public Hashtable context {get;set;}

        public static Random randomGenerator = new Random();
        public static RandomBeta betavariate = new RandomBeta();

        public abstract void reset(Hashtable extraInfo);

        public abstract VectorD getState();
        public abstract void step(Hashtable context);


        public EnvironmentRL(string name, TypeExecution typeExecution=TypeExecution.train, int patience_max=-1, int verbose_level=0, bool auto_increase_timestep=true){


            this.name = name;
            this.setTypeExecution(typeExecution);
            this.reward = 0.0f;
            this.game_over = false;
            this.verbose_level = verbose_level;
            this.patience = 0;
            this.patience_max = patience_max;
            this.reward_best_episode = float.MinValue;
            this.auto_increase_timestep = auto_increase_timestep;

        }

        public void setTypeExecution(TypeExecution typeExecution){

            this.typeExecution = typeExecution;
            is_validation_or_test = typeExecution != TypeExecution.train;
            is_train_only = typeExecution == TypeExecution.train;
            is_validation_only = typeExecution == TypeExecution.validation;
            is_test_only = typeExecution == TypeExecution.test;
        }

        public void setExtraContext(Hashtable context){
            this.context = context;
        }

        public bool isEarlyStopping(float reward_episode){
            bool early_stopping = false;

            if(this.patience_max > 0){
                if (reward_episode <= this.reward_best_episode)
                    this.patience += 1;
                else if (reward_episode > this.reward_best_episode){
                    this.reward_best_episode = reward_episode;
                    this.patience = 0;
                }
                    
                if (this.patience >= this.patience_max){
                    early_stopping = true;
                }
            }

            if(reward_episode > 10000){
                early_stopping = true;
            }


            return early_stopping;
        }

        
           
        
    }

    public class EnvironmentCommon:EnvironmentRL
    {
        public EnvironmentCommon(
            string name,
            TypeExecution typeExecution=TypeExecution.train,
            int patience_max=-1,
            int verbose_level=1
        ):base(name,typeExecution,patience_max,verbose_level){

            this.reset(new Hashtable{});
        }

        public override void reset(Hashtable extraInfo){
            this.reward = 0.0f;
            this.game_over = false;
            this.timestep = 0;

        }

        public override VectorD getState(){
            // state defined outside
            return new VectorD();
        }

        public override void step(Hashtable context)
        {
            throw new NotImplementedException();
        }
    }
}
