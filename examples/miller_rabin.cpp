#include <skepu>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>

static long pow_mod(long a, long x, long n)
{
  long r = 1;
  while (x)
  {
    if ((x & 1) == 1)
      r = a * r % n;
    x >>= 1;
    a = a * a % n;
  }
  return r;
}

#define DEFAULT_ACCURACY 5

int isprime(skepu::Index1D index, skepu::Random<DEFAULT_ACCURACY> &prng, int k)
{
  // Remap from [0 n) to [2 n+2)
  int n = index.i + 2;
  bool definitely_prime = false;
  bool definitely_not_prime = false;
  bool factor_found = false;
  
  // Must have ODD n greater than THREE
  if (n == 2 || n == 3) definitely_prime = true;
  if (n <= 1 || !(n & 1)) definitely_not_prime = true;

  // Write n-1 as d*2^s by factoring powers of 2 from n-1
  int s = 0;
  for (long m = n-1; !(m & 1); ++s, m >>= 1)
    ; // loop

  long d = (n - 1) / (1 << s);

  for (int i = 0; i < k; ++i)
  {
    long a = (prng.get() % (n-3)) + 2; //rand_between(2, n - 2); IMPORTANT
    if (definitely_prime || definitely_not_prime) continue;
    long x = pow_mod(a, d, n);
    
    if (x == 1 || x == n - 1)
      continue;
    
    for (int r = 1; r <= s - 1; ++r)
    {
      x = pow_mod(x, 2, n);
      if (x == 1) factor_found = true;
      if (x == n - 1) goto LOOP;
    }
    
    factor_found = true;
LOOP:
    continue;
  }

  // n is *probably* prime
  if (definitely_prime) return true;
  if (definitely_not_prime) return false;
  return !factor_found;
}


/*
 * Return the number of primes less than or equal to n, by virtue of brute
 * force.  There are much faster ways of computing this number, but we'll
 * use it to test the primality function.
 *
 */
static int pi(int n)
{
  auto prime_counter = skepu::MapReduce<0>(isprime, skepu::util::add<int>);
  prime_counter.setDefaultSize(n - 2);
  
  skepu::PRNG prng;
  prime_counter.setPRNG(prng);
  
  int r = prime_counter(DEFAULT_ACCURACY);

  return r;
}

int main(int argc, char *argv[])
{
  if (argc < 2)
	{
		skepu::io::cout << "Usage: " << argv[0] << " backend\n";
		exit(1);
	}
  
	auto spec = skepu::BackendSpec{argv[1]};
	skepu::setGlobalBackendSpec(spec);
  
  int expected[] = {4, 25, 168, 1229, 9592, 78498, 664579};

  for (int n = 10, e = 0; n <= 10000000; n *= 10, ++e)
  {
    int primes = pi(n);
    skepu::io::cout << "There are " << primes << " primes less than " << n;

    if (primes == expected[e]) skepu::io::cout << "\n";
    else skepu::io::cout << " --- FAIL, expecteded " << expected[e] << "\n";
  }
  
  return 0;
}