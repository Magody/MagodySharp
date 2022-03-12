using System;

namespace custom_lib
{

	public class CustomRandom
	{
		const int MBIG = int.MaxValue;
		const int MSEED = 161803398;

		int inext, inextp;
		int [] SeedArray = new int [56];
		
		public CustomRandom (int Seed)
		{
			int ii;
			int mj, mk;

			// Numerical Recipes in C online @ http://www.library.cornell.edu/nr/bookcpdf/c7-1.pdf

			// Math.Abs throws on Int32.MinValue, so we need to work around that case.
			// Fixes: 605797
			if (Seed == System.Int32.MinValue)
				mj = MSEED - (int)Math.Abs ((float)System.Int32.MinValue + 1f);
			else
				mj = MSEED - (int)Math.Abs ((float)Seed);
			
			SeedArray [55] = mj;
			mk = 1;
			for (int i = 1; i < 55; i++) {  //  [1, 55] is special (Knuth)
				ii = (21 * i) % 55;
				SeedArray [ii] = mk;
				mk = mj - mk;
				if (mk < 0)
					mk += MBIG;
				mj = SeedArray [ii];
			}
			for (int k = 1; k < 5; k++) {
				for (int i = 1; i < 56; i++) {
					SeedArray [i] -= SeedArray [1 + (i + 30) % 55];
					if (SeedArray [i] < 0)
						SeedArray [i] += MBIG;
				}
			}
			inext = 0;
			inextp = 31;
		}

		protected virtual double Sample ()
		{
			int retVal;

			if (++inext  >= 56) inext  = 1;
			if (++inextp >= 56) inextp = 1;

			retVal = SeedArray [inext] - SeedArray [inextp];

			if (retVal < 0)
				retVal += MBIG;

			SeedArray [inext] = retVal;

			return retVal * (1.0 / MBIG);
		}

		public virtual int Next ()
		{
			return (int)(Sample () * int.MaxValue);
		}

		public virtual int Next (int maxValue)
		{
			return (int)(Sample () * maxValue);
		}

		public virtual int Next (int minValue, int maxValue)
		{
			
			// special case: a difference of one (or less) will always return the minimum
			// e.g. -1,-1 or -1,0 will always return -1
			uint diff = (uint) (maxValue - minValue);
			if (diff <= 1)
				return minValue;

			return (int)((uint)(Sample () * diff) + minValue);
		}

		public virtual void NextBytes (byte [] buffer)
		{

			for (int i = 0; i < buffer.Length; i++) {
				buffer [i] = (byte)(Sample () * (byte.MaxValue + 1)); 
			}
		}

		public virtual double NextDouble ()
		{
			return this.Sample ();
		}
	}
}