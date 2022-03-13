using System;
using System.Collections;
using System.Collections.Generic;

using types;
using custom_lib;

using reinforcement_learning;

namespace test
{
    public class EnvironmentCartPole : EnvironmentRL
    {
        public static float NULL_VALUE = -1114242f;
        float gravity = 9.8f;  // acceleration due to gravity, positive is downward, m/sec^2
        float mcart = 1.0f;  // cart mass in kg
        float mpole = 0.1f;  // pole mass in kg
        float lpole = 0.5f;  // half the pole length in meters

        public float simulation_time_seconds = 60f;
           
        public float simulation_time_step = 0.01f;  // time step in seconds
        public int simulation_max_steps = 1;

        public float position_limit {get;set;}
        public float angle_limit_radians {get;set;}

        public float t;
        public float x;
        public float theta;
        public float dx;
        public float dtheta;
        public float xacc;
        public float tacc;


        float base_x;
        float base_theta;
        float base_dx;
        float base_dtheta;

        public EnvironmentCartPole(
            string name, 
            TypeExecution typeExecution, 
            int patience_max=-1, 
            int verbose_level=0,
            float x=-1114242f, 
            float theta=-1114242f, 
            float dx=-1114242f, 
            float dtheta=-1114242f,
            float position_limit=2.4f, 
            float angle_limit_radians=45 * MathF.PI / 180,
            bool auto_increase_timestep=false
        ) : base(name, typeExecution, patience_max, verbose_level, auto_increase_timestep)
        {

            this.simulation_max_steps = (int)(simulation_time_seconds/simulation_time_step);

            this.position_limit = position_limit;
            this.angle_limit_radians = angle_limit_radians;

            this.base_x = x;
            this.base_theta = theta;
            this.base_dx = dx;
            this.base_dtheta = dtheta;


            this.reset(new Hashtable{});


        }

        public static bool isNull(float value){
            return value == EnvironmentCartPole.NULL_VALUE;
        }


        /// <summary>
        /// Method <c>step</c> Update the system state using leapfrog integration.
        /// x_{i+1} = x_i + v_i * dt + 0.5 * a_i * dt^2
        /// v_{i+1} = v_i + 0.5 * (a_i + a_{i+1}) * dt
        /// </summary>
        public override void step(Hashtable context){

            float force = (float)context["force"];

            // Locals for readability.
            float g = this.gravity;
            float mp = this.mpole;
            float mc = this.mcart;
            float mt = mp + mc;
            float L = this.lpole;
            float dt = this.simulation_time_step;
            // acelerations actual
            float tacc0 = this.tacc;
            float xacc0 = this.xacc;


            // Making my action to have influence in the next state
            // Compute new accelerations as given in "Correct equations for the dynamics of the cart-pole system"
            // by Razvan V. Florian (http://florian.io).
            // http://coneural.org/florian/papers/05_cart_pole.pdf
            float st = MathF.Sin(this.theta);
            float ct = MathF.Cos(this.theta);
            float tacc1 = (g * st + ct * (-force - mp * L * MathF.Pow(this.dtheta, 2) * st) / mt) / (L * (4.0f / 3 - mp * MathF.Pow(ct,2) / mt));
            float xacc1 = (force + mp * L * (MathF.Pow(this.dtheta, 2) * st - tacc1 * ct)) / mt;

            // Update velocities.
            this.dx += 0.5f * (xacc0 + xacc1) * dt;
            this.dtheta += 0.5f * (tacc0 + tacc1) * dt;

            // Update position/angle.
            this.x += (dt * this.dx) + (0.5f * xacc1) * (float)MathF.Pow(dt,2);
            this.theta += (dt * this.dtheta) + (0.5f * tacc1) * MathF.Pow(dt,2);

            
            this.timestep += dt;
        }
        
        /// <summary>
        /// Method <c>getState</c> Get full state, scaled into (approximately) [0, 1].
        /// </summary>
        public override VectorD getState()
        {
            return new VectorD{
                0.5 * (this.x + this.position_limit) / this.position_limit,
                (this.dx + 0.75) / 1.5,
                0.5 * (this.theta + this.angle_limit_radians) / this.angle_limit_radians,
                (this.dtheta + 1.0) / 2.0
            };
        }

        public override void reset(Hashtable extraInfo)
        {

            this.reward = 0.0f;
            this.game_over = false;
            this.timestep = 0;

            
            float x = this.base_x;
            float theta = this.base_theta;
            float dx = this.base_dx;
            float dtheta = this.base_dtheta;
            
            if(EnvironmentCartPole.isNull(x)){
                x = MathUtil.getRandomUniform(-0.5f * this.position_limit, 0.5f * this.position_limit);
            }

            if(EnvironmentCartPole.isNull(theta)){
                theta = MathUtil.getRandomUniform(-0.5f * this.angle_limit_radians, 0.5f * this.angle_limit_radians);
            }

            if(EnvironmentCartPole.isNull(dx)){
                dx = MathUtil.getRandomUniform(-1.0f, 1.0f);
            }

            if(EnvironmentCartPole.isNull(dtheta)){
                dtheta = MathUtil.getRandomUniform(-1.0f, 1.0f);
            }
                

            
            this.t = 0.0f;
            this.x = x;
            this.theta = theta;

            this.dx = dx;
            this.dtheta = dtheta;

            this.xacc = 0.0f;
            this.tacc = 0.0f;



        }


        public static float continuous_actuator_force(float action_as_prob){
            return -10.0f + 2.0f * action_as_prob;
        }

        public static float noisy_continuous_actuator_force(float action_as_prob){
            double a = action_as_prob + MathUtil.getRandomNormal(0, 0.2f);
            return ((float)a > 0.5)? 10.0f:-10.0f;
        }

        /// <summary>
        /// Method <c>discrete_actuator_force</c> param action: 0	Push cart to the left, 1 Push cart to the right
        /// </summary>
        public static float discrete_actuator_force(int action){
            return (action == 1)? 10.0f:-10.0f;
        }

        public static float noisy_discrete_actuator_force(int action_as_prob){
            double a = action_as_prob + MathUtil.getRandomNormal(0, 0.2f);
            return ((float)a > 0.5)? 10.0f:-10.0f;
        }

    }
    

}