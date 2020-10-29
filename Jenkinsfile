def FAILURE_STAGE
pipeline {
   agent any

   stages {
      stage('Clone repo') {
         steps {
            script{
                FAILURE_STAGE=env.STAGE_NAME
            }
            cleanWs()
            sh 'git clone --recursive https://oauth2:BkY9r8zsTGBCFBVDNPy3@gitlab.seis.exa2pro.iti.gr/exa2pro/skepu-clang.git'
         }
      }
      stage('Build') {
         steps {
            script{
                FAILURE_STAGE=env.STAGE_NAME
            }
            sh '''
	            cd ./skepu-clang
	            mkdir build
	            cd build
        	    cmake .. -DCMAKE_BUILD_TYPE=Release -DSKEPU_ENABLE_TESTING=ON
	            make -j4
                '''
         }
      }
      stage('Test') {
         steps {
            script{
                FAILURE_STAGE=env.STAGE_NAME
            }
            sh '''
                find skepu-headers/tests -type f -exec sed -i 's:mpiexec":mpiexec" "--allow-run-as-root":' {} \;
                ctest -VV
                '''
         }
      }
   }
}