using System;

public class RandomBeta
{
  // the "BA" algorithm
  // R.C.H. Cheng
  public Random rnd;
  public RandomBeta(int seed=42) {
    this.rnd = new Random(seed);
  }
  public double sample(double a, double b) {
    // a, b greater-than 0
    double alpha = a + b;
    double beta = 0.0;
    double u1, u2, w, v = 0.0;

    if (Math.Min(a, b) <= 1.0)
      beta = Math.Max(1 / a, 1 / b);
    else
      beta = Math.Sqrt((alpha - 2.0) /
        (2 * a * b - alpha));
    double gamma = a + 1 / beta;

    while (true) {  // no! add a sanity counter!
      u1 = this.rnd.NextDouble();
      u2 = this.rnd.NextDouble();
      v = beta * Math.Log(u1 / (1 - u1));
      w = a * Math.Exp(v);
      double tmp = Math.Log(alpha / (b + w));
      if (alpha * tmp + (gamma * v) - 1.3862944 >= Math.Log(u1 * u1 * u2))
        break;
    }

    double x = w / (b + w);
    return x;
  }
}