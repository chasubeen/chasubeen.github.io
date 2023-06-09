---
layout: single
title:  "[ECC Github 스터디 3주차] 4강. 깃허브 시작하기"
categories: Git/Github
tags: [ECC, Git/Github] 
author_profile: false
---

# **1. 원격 저장소와 깃허브**

## **1-1. 원격 저장소란**
- 지역 저장소(local rwepository): 개인 컴퓨터 내의 저장소
- 원격 저장소(remote repository)
  - 지역 저장소가 아닌 컴퓨터나 서버에 만든 저장소
  - 깃에서는 지역 저장소와 원격 저장소를 연결해서 버전을 관리하는 파일을 쉽게 백업할 수 있음
  - **백업**과 **협업**에 중요한 역할을 수행
  - 깃허브는 원격 저장소를 제공하는 서비스

## **1-2. 깃허브로 할 수 있는 일들**
1. 원격 저장소에서 깃을 사용할 수 있다.
  - 깃허브는 깃 사용을 위한 **원격 저장소**를 제공하는 서비스 -> 온라인에서 깃의 버전 관리 기능을 사용할 수 있음
  - 지역 저장소를 만들지 않아도 깃허브에 원격 저장소를 만들어 사용할 수 있음
  - 지역 저장소가 있다면 원격 저장소와 연결해서 사용할 수도 있음
2. 지역 저장소를 백업할 수 있다.
  - 깃허브에 원격 저장소를 만들고 사용자 컴퓨터의 지역 저장소를 연결한 후 동기화하면 지역 저장소를 인터넷에서 백업할 수 있음
  - 깃허브에 백업 시 원격 저장소에 쉽게 **커밋**할 수 있음 
3. 온라인 개발 툴을 사용할 수 있다.
  - **코드스페이스(Codespace)** 기능
  - 깃허브를 통해 나만의 개발 환경을 만들어 놓을 수 있고, 언제든지 온라인에서 VS Code 편집기를 열어 수정/커밋 가능
    - 지역 저장소를 만들고 깃허브로 올리는(**push**) 과정도 필요 x 
4. 협업 프로젝트에 사용할 수 있다.
  - 원격 저장소이기에 누구나 접근할 수 있음
  - 깃과 깃허브에서 여러 가지 협업 도구를 제공함
5. 자신의 개발 이력을 남길 수 있다.
  - 깃허브에서 소스를 수정하고 오픈 소스에 참여해서 하는 일은 사용자 초기 화면에 날짜별로 모두 기록으로 남음
6. 다른 사람의 소스를 살펴볼 수 있고, 오픈 소스에 참여할 수도 있다.
  - 다른 개발자들이 공개해 놓은 소스들을 얼마든지 내 저장소로 가져와 분석해 볼 수 있음
  - 깃을 비롯해 웹 개발이나 인공지능, 데이터 과학 등 전 개발 분야에 걸쳐 다양한 오픈 소스가 등록되어 있음

# **2. 깃허브 가입하기**
- ~~이 문서를 깃허브 내의 markdown으로 작성했는데..^^?~~

## **2-1. 깃허브에 가입하기**
~~- 이미 되어있다.~~
- 그 대신 학생 pro 지원을 받았다 ^-^

## **2-2. 지역 저장소와 원격 저장소**
- ```푸시(push)```: 지역 저장소에서 원격 저장소로 커밋을 등록하는 것(지역 -> 원격)
- ```풀(pull)```: 원격 저장소의 변경 사항을 지역 저장소로 내려받는 것(원격 -> 지역)
- ```동기화(synchronize)```: 지역 저장소와 원격 저장소를 항상 같게 유지하는 것 

## **2-3. 원격 저장소 만들기**
![image](https://user-images.githubusercontent.com/98953721/228867714-0a8ebc92-e4f8-441b-a4eb-b3c64fb779d4.png)

- 원격 저장소의 HTTPS 주소 형식
```
https://github.com/아이디/저장소명
```

# **3. 지역 저장소를 원격 저장소에 연결하기**

## **3-1. 지역 저장소 만들기**
- git bash를 활용하여 지역 저장소를 생성

## **3-2. 원격 저장소에 연결하기**
### **● 커맨드 라인에서 기존 저장소를 푸시하기**
1. 지역 저장소와 연결할 원격 저장소 찾기 -> 깃허브 주소 복사하기
![image](https://user-images.githubusercontent.com/98953721/228881406-6eb2608b-d205-4da8-92a7-550668b4a80c.png)

2. 원격 저장소에 origin 추가하기(지역 저장소를 원격 저장소에 연결)
```
$ git remote add origin 복사한 깃허브 주소
```

- origin: 깃허브 저장소 주소
- remote: 원격 저장소

![image](https://user-images.githubusercontent.com/98953721/228892185-07f5ec75-1c9f-41df-90b3-0eee608fb8e8.png)

3. 원격 저장소에 제대로 연결되었는지 확인하기
```
$ git remote -v
```

![image](https://user-images.githubusercontent.com/98953721/228892306-81ee0d00-7b7a-49c3-9881-7af27d7f93a5.png)

# **4. 지역 저장소와 원격 저장소 동기화하기**

## **4-1. 원격 저장소에 커밋 올리기**
1. 지역 저장소의 브랜치를 origin(원격 저장소)의 main 브랜치로 푸시
```
$ git push -u origin main 
```

- 처음으로 원격 저장소에 푸시할 때는 깃허브 계정 사용자 인증이 필요함
- 사용자 인증이 끝나는것과 동시에 터미널 창에서는 푸시가 진행됨

![image](https://user-images.githubusercontent.com/98953721/228892467-70fb2cad-1163-4419-9c58-826f6b006fd8.png)  

**🔺에러 해결**  
- 왠만하면 빈 repo 하나 만들어서 실습하세요.
- ```-u``` 옵션 때문에 사용자 인증이랑 같이 들어가는 과정에서 충돌이 발생한 듯 합니다.
  - 빈 repo가 아니라서 지역 저장소랑 원격 저장소랑 동기화가 안 된 상태
  - ~~git pull도 해봤는데..안먹히더라고요..?!~~

![image](https://user-images.githubusercontent.com/98953721/228892991-87fc62b9-1028-466a-9eab-95bdaebd2280.png)  

- 지역 저장소에 있던 f1.txt 파일이 원격 저장소에 올라감
- 커밋 로그를 통해 커밋한 날짜와 사람, 메시지 등을 확인할 수 있음

![image](https://user-images.githubusercontent.com/98953721/228893742-e2b37c17-9383-458e-a081-65d7e9fe3f9d.png)

## **4-2. 원격 저장소에 파일 올리기**
- ```git push``` 명령어 활용

![image](https://user-images.githubusercontent.com/98953721/228894669-aa8a2515-e07a-4953-8b5b-c4fbee269389.png)

- 깃허브에 commit log가 추가된 것을 확인할 수 있음

![image](https://user-images.githubusercontent.com/98953721/228894988-11fe33d0-50ac-4550-9c12-4dda605be052.png)

- 해당 커밋으로 변경된 내용을 구체적으로 확인하기 위해서는 커밋 이름 오른쪽에 있는 **커밋 해시**를 클릭하여 확인 
![image](https://user-images.githubusercontent.com/98953721/228895537-cee344aa-ac5c-496d-aedb-2a3c0406ca0f.png)

## **4-3. 원격 저장소에서 직접 커밋하기**
- 원격 저장소에 접속해서 커밋을 만들면 지역 저장소에는 없기 때문에 원격 저장소에서 지역 저장소로 가져와서 **동기화**를 진행해야 함

## **4-4. 원격 저장소에서 커밋 내려받기**
- 원격 저장소에 있는 소스 파일을 다른 사용자가 수정하거나 깃허브 사이트에서 직접 커밋하면 지역 저장소와 **버전 차이**가 발생함
  - 원격 저장소와 지역 저장소의 상태를 같게 만들기 위해 원격 저장소의 커밋을 지역 저장소로 가져옴 => ```풀(pull)```
- 명령어
```
$ git pull origin main
```
![image](https://user-images.githubusercontent.com/98953721/229121563-7a25b710-4f43-40eb-94e6-8da17ae349fb.png)

- 커밋 로그 확인
![image](https://user-images.githubusercontent.com/98953721/229122420-de720db3-6f1a-4744-ab6d-e4d599c13e73.png)

## **4-5. 깃허브 원격 저장소 화면 살펴보기**
- 교재 p.164


# **5. 깃허브에 SSH 원격 접속하기**

## **5-1. SSH 원격 접속**
### **● SSH란?**
- secure shell의 줄임말로, 보안이 강화된 안전한 방법으로 정보를 교환하는 방식
- 기본적으로 프라이빗 키(private key)와 퍼블릭 키(public key)를 한 쌍으로 묶어 컴퓨터를 인증

## **5-2. SSH 키 생성**
1. 홈 디렉터리로 이동 -> SSH가 저장되는 디렉터리 확인하기
![image](https://user-images.githubusercontent.com/98953721/229123642-231993a3-ff1d-4dde-a250-79f52d91590d.png)

2. ```Enter``` 키를 두 번 더 눌러 SSH 접속을 위한 비밀번호 생성
  - ```id_rsa```: 프라이빗 키
  - ```id_rsa.pub```: 퍼블릭 키

3. 저장된 키 확인하기

![image](https://user-images.githubusercontent.com/98953721/229124349-1e4263a3-a4c5-403f-9876-bbf9b7eb4864.png)

## **5-3. 깃허브에 퍼블릭 키 전송하기**
1. 사용자 컴퓨터에 만들어져 있는 퍼블릭 키를 깃허브 서버로 전송한 다음 저장
- ```id_rsa.pub``` 파일의 내용 확인하기

![image](https://user-images.githubusercontent.com/98953721/229126174-c8fbe251-9e03-495d-a247-6a757c90cefb.png)

2. 사용자 컴퓨터에 있는 **프라이빗 키**와 깃허브 서버에 있는 **퍼블릭 키**를 비교
  - 두 키가 서로 맞으면 사용자 컴퓨터와 깃허브 저장소가 연결됨

- 이후 과정: 교재 p.170 ~ 171

## **5-4. SSH 주소로 원격 저장소 연결하기**
- SSH 주소를 사용해 원격 저장소에 연결

```
$ git remote add origin 복사한 ssh 주소
```

- 연결된 원격 저장소 확인

```
$ git remote -v
```

![image](https://user-images.githubusercontent.com/98953721/229128535-88480e5b-1d10-4efe-9b4b-7129d49d383b.png)

- 원격 저장소로 푸시

```
$ git push -u origin main
```

![image](https://user-images.githubusercontent.com/98953721/229129840-7623c874-afa3-4036-a767-9d34a3cc7ea0.png)

---
# **📚 References**
Do it! 지옥에서 온 문서 관리자 깃&깃허브 입문_4장
