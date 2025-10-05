import java.text.SimpleDateFormat
def TODAY = (new SimpleDateFormat("yyyyMMddHHmmss")).format(new Date())
pipeline {
    agent any
    environment {
        strDockerTag = "${TODAY}_${BUILD_ID}"
        strDockerImage ="530hyelim/cicd_ypjp_bot:${strDockerTag}"
        strGitUrl = "https://github.com/kh-powerpuffgirls/yoripick-joripick-bot.git"
    }
    stages {
        // 1. 깃헙 체크아웃(master)
        stage('Checkout') {
            steps {
                git branch: 'jenkins', url: strGitUrl
            }
        }
        // 2. 도커 이미지 빌드
        stage('Docker Image Build') {
            steps {
                script {
                    echo "🔧 Building Python Docker image..."
                    oDockImage = docker.build(strDockerImage, "-f Dockerfile .")
                }
            }
        }
        // 3. 도커 이미지 푸쉬
        stage('Docker Image Push') {
            steps {
                script {
                    docker.withRegistry('', 'Dockerhub_Cred') {
                        echo "🚀 Pushing Docker image to Docker Hub..."
                        oDockImage.push()
                    }
                }
            }
        }
        // 4. 프로덕션 서버 배포
        stage('Deploy Production') {
            steps {
                sshagent(credentials: ['SSH-PrivateKey']) {
                    echo "☁️ Deploying to EC2..."
                    sh "ssh -o StrictHostKeyChecking=no ec2-user@3.38.116.62 docker container rm -f ypjp-bot"
                    sh "ssh -o StrictHostKeyChecking=no ec2-user@3.38.116.62 docker container run \
                        -d \
                        -p 8000:8000 \
                        --name=ypjp-bot \
                        -v /etc/letsencrypt/archive/bot.ypjp.store/fullchain1.pem:/app/certs/fullchain.pem:ro \
                        -v /etc/letsencrypt/archive/bot.ypjp.store/privkey1.pem:/app/certs/privkey.pem:ro \
                        ${strDockerImage} \
                        uvicorn faqService:app --host 0.0.0.0 --port 8000 \
                        --ssl-certfile /app/certs/fullchain.pem \
                        --ssl-keyfile /app/certs/privkey.pem"
                }
            }
        }
    }
}
