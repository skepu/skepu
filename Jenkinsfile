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
   post {
        success {
            mail bcc: '', body: "<b>Build status: Success</b><br>Jenkins pipeline: ${env.JOB_NAME} <br>Build Number: ${env.BUILD_NUMBER} <br> GitLab project: skepu-clang <br> Branch: master", cc: '', from: 'jenkins@gitlab.seis.iti.gr', replyTo: '', subject: "JENKINS EMAIL NOTIFICATION: Project name -> ${env.JOB_NAME}", to: 'theioak@iti.gr,johan.ahlqvist@liu.se,august.ernstsson@liu.se', charset: 'UTF-8', mimeType: 'text/html'
        }
        failure {
            mail bcc: '', body: "<b>Build status: Failed</b><br>Jenkins pipeline: ${env.JOB_NAME} <br>Build Number: ${env.BUILD_NUMBER} <br> GitLab project: skepu-clang <br> Branch: master", cc: '', charset: 'UTF-8', from: '', mimeType: 'text/html', replyTo: '', subject: "JENKINS EMAIL NOTIFICATION (CI ERROR): Project name -> ${env.JOB_NAME}", to: "theioak@iti.gr,johan.ahlqvist@liu.se,august.ernstsson@liu.se"
        }
   }
}