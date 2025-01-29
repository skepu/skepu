
#include <skepu>
#include <skepu-lib/io.hpp>


float square(float elem)
{
  return elem * elem;
}

float addone(float elem)
{
  return elem + 1;
}


int main()
{
//  skepu::VectorStream<float> myleftinstream(istream);  
//  skepu::VectorStream<float> myrightinstream(istream);  
//  skepu::VectorStream<float> myoutstream(ostream);

  size_t N = 1000;

  skepu::Vector<float> myinstream(N);
  skepu::Vector<float> myoutstream(N);
  
  for (size_t i = 0; i < N; ++i)
  {
    myoutstream(i) = i;
  }
  
  auto stage1 = skepu::Map(square);
  auto stage2 = skepu::Map(addone);
  auto stage3 = skepu::Map(addone);
  auto pipeline = skepu::Pipe(stage1, stage2); // stage1 -> stage2
//  auto pipeline2 = pipeline >> stage3;
  
  auto pipeline2 = stage1 >> stage2 >> stage3;
  
//  auto pipelineB = skepu::Pipe(square, addone); // derive from function signature (look for Region etc.)
  
//  auto pipelineC = skepu::Pipe(skepu::Map(square), skepu::Map(addone), ...); // explicit "decorator"
  
//  auto stage1 = skepu::Map(square);
//  auto stage2 = skepu::Map(addone);
//  auto pipelineD = stage1 >> stage2;
  
  
//  auto farmD = taskA || taskB;
  
  
  pipeline2(myoutstream, myoutstream);
  
  
  skepu::io::cout << myoutstream << "\n";

}


/*
{
  skepu::pipeline(myleftinstream, myrightinstream,
  [](float elem) {
    return elem * elem;
  },
  [] (float elem) {
    return elem + 1;
  },
  outstream1, out2);
  
  
  
  
}*/