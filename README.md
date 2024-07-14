# Description
The project is a recommendation system of text posts based on data about users, 
publications and interactions between them.

# Run the Code
1. You need to copy the repository to yourself locally:
```
git clone https://github.com/zayolya/ml.git
```

2. Create an image of the Docker container:
```
docker build . -t TAG
```
where TAG is its name.

3. Create a Docker container:
```
docker run -p 8081:8081 TAG
```
where TAG is the name from the previous step. 

4. After the time allotted to start the service (to load the initial databases), it will become available.
You can check the operation at the local address, which starts with:
```
http://127.0.0.1:8081/
```
or
```
http://localhost:8081/
```
5. For example, query 5 (limit) recommendations for user 3089 (id) 
at time 2021-12-23 (time) looks like this
```
http://localhost:8081/post/recommendations/?id=3089&time=2021-12-23 12:00:01&limit=5
```

![Screenshot 2024-07-14 at 11.40.12.png](screenshot%2FScreenshot%202024-07-14%20at%2011.40.12.png)