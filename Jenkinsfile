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
            sh 'git clone --recursive https://oauth2:3bHxZoCBz8hC8QJKz4u7@gitlab.seis.exa2pro.iti.gr/exa2pro/skepu-clang.git'
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
        	    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/skepu-clang ..
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
                pwd
                ls
                ls skepu-clang/build/llvm/bin/
                cd skepu-clang
                git branch
                git log
                '''
         }
      }
   }
   post {
        success {
            mail bcc: '', body: "<b>Build status: Success</b><br>Jenkins pipeline: ${env.JOB_NAME} <br>Build Number: ${env.BUILD_NUMBER} <br> GitLab project: skepu-clang <br> Branch: master", cc: '', from: 'jenkins@gitlab.seis.exa2pro.iti.gr', replyTo: '', subject: "JENKINS EMAIL NOTIFICATION: Project name -> ${env.JOB_NAME}", to: 'theioak@iti.gr,johan.ahlqvist@liu.se,august.ernstsson@liu.se', charset: 'UTF-8', mimeType: 'text/html'
        }
        failure {
            mail bcc: '', body: "<b>Build status: Failed</b><br>Jenkins pipeline: ${env.JOB_NAME} <br>Build Number: ${env.BUILD_NUMBER} <br> GitLab project: skepu-clang <br> Branch: master", cc: '', charset: 'UTF-8', from: 'jenkins@gitlab.seis.exa2pro.iti.gr', mimeType: 'text/html', replyTo: '', subject: "JENKINS EMAIL NOTIFICATION (CI ERROR): Project name -> ${env.JOB_NAME}", to: "theioak@iti.gr,johan.ahlqvist@liu.se,august.ernstsson@liu.se"
        }
   }
}