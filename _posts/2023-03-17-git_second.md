---
layout: single
title:  "[ECC Github 스터디 1주차] 2강. 깃으로 버전 관리하기"
categories: Git/Github
tags: [ECC, Git/Github] 
author_profile: false
---

# **1. 깃 저장소 만들기**
- 저장소(repository): 폴더 안에 버전이 저장되는 공간
- 현재 디렉터리 초기화
  - ```git init``` 명령
  - 'Initialized empty Git repository ...'라는 메시지가 나타나면 해당 디렉토리에서 깃 사용 가능
  - ```(main)```: 깃을 위한 저장소가 생성됨을 의미
  - ```.git/``` 디렉터리가 생성됨 -> 깃의 버전이 저장될 **저장소**

# **2. 버전 만들기**
- 버전: 번호 등을 통해 구별하는 수정 내용 등을 통칭

## **2-1. 깃에서 버전이란**
- 문서를 수정하고 저장할 때마다 생기는 것
- 깃에서 버전을 관리하면 원래 파일 이름은 그대로 유지하면서 파일에서 무엇을 변경했는지를 변경 시점마다 저장 가능
- 버전마다 작업한 내용을 확인 가능, 버전 되돌리기 가능

## **2-2. 스테이지와 커밋 이해하기**
- 작업 트리 -> 스테이징 영역 -> 저장소
![image](https://user-images.githubusercontent.com/98953721/225901410-259738a0-92f4-489e-9c25-0c6203988f58.png)

### **● 작업 트리(working tree)**
- 파일 수정, 저장 등의 작업을 하는 디렉터리 -> 작업 디렉터리라고도 함
- 우리 눈에 보이는 디렉터리
### **● 스테이지**
- 버전으로 만들 파일이 대기하는 곳 -> 스테이징 영역이라고도 함
### **● 저장소(reposiroty)**
- 스테이지에 대기하고 있던 파일들을 버전으로 만들어 저장하는 곳

## **2-3. 작업 트리에서 문서 수정하기**
- ```git status```: 현재 깃 상태 확인

![image](https://user-images.githubusercontent.com/98953721/225901818-95081ac4-c11b-45c3-9fe2-747253992c84.png)

- On branch main: 현재 **main** branch에 있는 상태
- No commits yet: 아직 커밋한 파일이 x
- nothing to commit: 현재 커밋할 파일이 x

![image](https://user-images.githubusercontent.com/98953721/225903083-3075b7b9-b8d0-4688-a7c5-860a4cc44068.png)  
- untracked file: 버전을 한 번도 관리하지 않은 파일

## **2-4. 수정한 파일을 스테이지에 올리기**
- **스테이징(staging)**
  - 깃에게 버전 만들 준비를 하라고 알려 주는 것
  - 작업 트리 -> 스테이지 
- ```git add``` 명령어 활용  

![image](https://user-images.githubusercontent.com/98953721/225904524-9b6196eb-4290-4774-93eb-a150ab89200a.png)
- '새 파일 hello.txt를 앞으로 커밋할 것이다.' 

## **2-5. 스테이징한 파일 커밋하기**
- **커밋(commit)**  
  - 깃에서 버전을 만드는 과정
  - 버전의 변경 사항을 확인할 수 있도록 메시지를 함께 기록해 두어야 함
- ```git commit``` 명령을 활용
  - ```-m``` 옵션: 커밋 메시지 작성
  - ```--amend``` 옵션: 최근에 커밋한 파일의 커밋 메시지 수정(덮어쓰기)
 
![image](https://user-images.githubusercontent.com/98953721/225906092-f6263e94-4eb7-45b6-ab44-a885f43f2cc6.png)
  - 파일 1개가 변경되었고(1 file changed), 파일에 1개의 내용이 추가되었음(1 insertion(+))
  - 스테이지 -> 저장소 
- ```git log```: 저장소에 저장된 버전 확인
  - 가장 최근 버전 정보가 최상단에 표시됨

![image](https://user-images.githubusercontent.com/98953721/225906521-73300040-6cad-48f0-b05d-36cb613d241f.png)

## **2-6. 스테이징과 커밋을 한 번에 처리하기**
- commit 명령에 ```-am``` 옵션 사용 시 스테이징과 커밋을 **한번에** 처리 가능
  - 단, 한 번이라도 커밋한 적이 있는 파일을 다시 커밋할 때만 사용 가능

![image](https://user-images.githubusercontent.com/98953721/225907513-4765f5ea-d30c-4645-a0fa-e0cfb898a3d4.png)
  - 스테이징과 커밋 과정이 한꺼번에 보이게 됨
 
# **3. 커밋 내용 확인하기**

## **3-1. 커밋 기록 확인**
- ```git log``` 명령 
  - 지금까지 만든 버전/ 설명 확인
  - ```--stat``` 옵션: 커밋과 관련된 파일까지 함께 확인

![image](https://user-images.githubusercontent.com/98953721/225908465-736ebf30-7e48-4f16-a872-38d158448d73.png)

  - **용어정리**
    - 커밋 해시(commit hash)
      - 커밋을 구별하는 아이디 정도
      - 커밋 해시 옆 ```(HEAD -> main)```은 해당 버전이 가장 **최신**임을 표시
    - 커밋 로그(commit log): git log 명령 입력 시 나오는 정보
  
## **3-2. 변경 사항 확인**
- ```git diff``` 명령
  - 작업 트리에 있는 파일과 스테이지에 있는 파일 비교
  - 스테이지에 있는 파일과 저장소에 있는 최신 커밋 비교  
  => 커밋 전 수정 파일을 최종  

![image](https://user-images.githubusercontent.com/98953721/225909902-95949de3-8cd1-43f0-8a2c-05bb5cb87997.png)
  - ```-2```: 삭제된 내용
  - ```+two```: 추가된 내용

# **4. 버전 만드는 단계마다 파일 상태 알아보기**

## **4-1. tracked 파일과 untracked 파일**
- tracked 파일
  - 깃은 한 번이라도 커밋한 파일은 계속해서 수정 사항이 있는지 추적
  - 깃이 추적하고 있는 파일
  - ```modified:```라고 상태가 표시됨
   
- untracked 파일
  - 한 번도 커밋하지 않은 파일
  - 수정 내용을 추적하지 않는 파일 
  - ```new file:```이라고 상태가 표시됨
  
- tracked, untracked 파일 모두 ```git add``` 명령어를 통해 작업 트리에서 스테이지에 올릴 수 있음

![image](https://user-images.githubusercontent.com/98953721/225911773-47d85ea2-4ef6-4a3f-b93e-08ca69814fde.png)

※ ```.gitignore``` 파일
- 한 디렉터리 안에 버전 관리를 하지 **않을** 파일/디렉터리가 섞여있는 경우 ```.gitignore``` 파일을 만들어 목록 지정 가능

## **4-2. unmodified, modified, stage 상태**
- **unmodified**  
  - 수정되지 않은 상태
  - 작업 트리에 아무 변경 사항이 없는 상태(working tree clean)
- **modified**  
  - 수정된 상태
  - changes not staged for commit: 파일이 수정만 된 상태, **작업 트리**
  - Changes to be committed: 커밋 직전 상태, **staged** 상태
  - 커밋 후 파일은 다시 수정 직전인 unmodified 상태로 돌아감

![image](https://user-images.githubusercontent.com/98953721/225916190-b7877067-b9f4-4f48-8f08-5360ec246b30.png)

# **5. 작업 되돌리기**
## **5-1. 작업 트리에서 수정한 파일 되돌리기**
- ```git restore``` 명령어: **작업 디렉터리**에서 수정한 내용 되돌리기
- ```cat``` 명령어를 통해 수정 내용이 정상적으로 지워졌는지 확인

## **5-2. 스테이징 되돌리기**
- 스테이징을 취소할 때도 ```git restore``` 명령어 사용
  - ```--staged``` 옵션: 스테이지에있는 모든 파일을 한꺼번에 되돌리기
  -  ```--staged 파일명```: 해당 파일만 골라서 되돌리기

## **5-3. 최신 커밋 되돌리기**
- 커밋을 취소하면 커밋과 스테이징이 함께 취소됨
- ```git reset``` 명령어 활용
  - ```HEAD^``` 옵션: 현재 HEAD가 가리키는 브랜치의 최신 커밋을 되돌림
  - 최신 커밋 취소, 스테이징 취소 -> 작업 트리에만 파일이 남음
  - 다른 옵션들  
  
  |명령|설명|
  |----|---------|
  |$git reset --soft HEAD^|커밋을 취소하고 파일을 staged 상태로 작업 디렉터리에 저장|
  |$git reset --mixed HEAD^|커밋을 취소하고 파일을 unstaged 상태로 작업 디렉터리에 보관|
  |$git reset HEAD^|--mixed 옵션을 사용할 때와 같이, 커밋을 취소하고 unstaged 상태로 작업 디렉터리에 보관|

## **5-4. 특정 커밋으로 되돌리기**
- ```git reset``` 명령 다음에 **커밋 해시**를 사용하여 특정 커밋으로 되돌릴 수 있음
  - 최근 커밋을 원하는 커밋으로 **리셋**하는 방식을 채택
-  ```git reset``` 명령어에서 ```--hard``` 옵션을 입력 후 커밋 해시를 붙여넣음
  - commit 후 해당 커밋 해시 위치로 HEAD가 옮겨짐
    - 해당 커밋이 가장 최신 커밋이 됨
    - 파일 내용 또한 되돌아 감

## **5-5. 커밋 변경 이력 취소**
- 변경 사항만 취소하고 커밋은 남겨두는 경우
- ```git revert``` 명령어 활용
```
$git revert 복사한 커밋 해시
```

- 커밋 메시지 입력 가능
- 커밋 메시지 맨 위에는 revert 한 버전이 표시됨
- 커밋을 취소하면서 남겨 둘 내용이 있다면 문서 맨 **위**에 입력 후 저장
- 기존 커밋이 사라지지 않고, 커밋을 취소했다는 내용이 **새로운** 커밋으로 추가됨
  - 변경 내용만 취소하고, 취소에 대한 커밋을 새로 생성















