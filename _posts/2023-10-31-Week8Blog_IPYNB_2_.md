---
toc: True
comments: True
layout: post
title: Images not rendering, How I fixed sqlite bug, Learning on Ports
type: tangibles
courses: {'csp': {'week': 8}}
---

# Troubleshooting Image Rendering Issues and Deploying on AWS

## Identifying the Problem
I recently encountered an issue where images weren't rendering properly on my web application. To tackle this problem, I decided to follow a systematic approach to identify and fix the issue.

## Step 1: Research and Understanding
I began by researching common causes of image rendering problems in web development. This involved studying the structure of image files, image formats, and how web browsers interpret and display images. Through this research, I gained insights into potential issues related to image paths, file formats, or server configurations.

## Step 2: Debugging Locally
I replicated the issue in my local development environment to understand the problem better. Using debugging tools, I inspected network requests, checked file paths, and verified image formats. This hands-on experience helped me pinpoint the root cause of the rendering problem.

## Step 3: Learning About Ports
During my investigation, I also delved into the concept of ports in networking. I realized that ports are essential for communication between different services within a computer network. Understanding ports enabled me to comprehend how my web application communicates with external resources and services.

## Step 4: Deploying on AWS
The knowledge of ports became crucial when deploying my application on AWS. I learned how to configure security groups and network ACLs to allow inbound and outbound traffic on specific ports. This understanding was pivotal in setting up my AWS environment correctly, ensuring that my application could communicate seamlessly with the necessary resources.

## Step 5: Future Steps
Moving forward, I plan to explore best practices for optimizing image loading in web applications. Additionally, I will continue to enhance my knowledge of networking and security concepts, enabling me to deploy more complex and secure applications on AWS.

By following this structured approach and deepening my understanding of both image rendering issues and networking concepts, I am confident in my ability to troubleshoot similar problems in the future and deploy applications successfully on AWS.

# Troubleshooting Image Rendering Issues, Understanding Ports, and Managing SQLite Databases

## Identifying the Problem
I recently encountered an issue where images weren't rendering properly on my web application. To tackle this problem, I decided to follow a systematic approach to identify and fix the issue.

## Step 1: Research and Understanding
I began by researching common causes of image rendering problems in web development. This involved studying the structure of image files, image formats, and how web browsers interpret and display images. Through this research, I gained insights into potential issues related to image paths, file formats, or server configurations.

## Step 2: Debugging Locally
I replicated the issue in my local development environment to understand the problem better. Using debugging tools, I inspected network requests, checked file paths, and verified image formats. This hands-on experience helped me pinpoint the root cause of the rendering problem.

## Step 3: Learning About Ports
During my investigation, I also delved into the concept of ports in networking. I realized that ports are essential for communication between different services within a computer network. Understanding ports enabled me to comprehend how my web application communicates with external resources and services.

## Step 4: Deploying on AWS
The knowledge of ports became crucial when deploying my application on AWS. I learned how to configure security groups and network ACLs to allow inbound and outbound traffic on specific ports. This understanding was pivotal in setting up my AWS environment correctly, ensuring that my application could communicate seamlessly with the necessary resources.

## Step 5: Managing SQLite Databases and Gitignore
In the process of developing my web application, I also encountered the challenge of managing SQLite databases. I discovered that these databases, containing critical application data, need to be included in the version control system. However, it's a common best practice not to include them in the repository directly. Instead, I learned to add `*.db` or the specific database file name to the `.gitignore` file. This ensures that sensitive database files are not accidentally shared on version control platforms like GitHub.

## Step 6: Future Steps
Moving forward, I plan to explore best practices for optimizing image loading in web applications. Additionally, I will continue to enhance my knowledge of networking and security concepts, enabling me to deploy more complex and secure applications on AWS. I'll also focus on database management strategies, including backup and migration techniques, to ensure the integrity of my application's data.

By following this structured approach and deepening my understanding of image rendering issues, networking concepts, and database management, I am confident in my ability to troubleshoot similar problems in the future, deploy applications successfully on AWS, and manage databases securely.

## Links
- [Commit 1](https://github.com/will-w-cheng/Frontend-influencer-innovator/commit/85a3429c5d264813555b3d25ebd7b24f0b97e5b2)
- [Commit 2](https://github.com/Saaras859/Team-Influencer-Innovators/commit/78986eaeb063379a19f9c6d10011cdd87a1f62f7)
- [Commit 3](https://github.com/Saaras859/Team-Influencer-Innovators/commit/5ebfcef15439ef72d2bbcad65041a7634011ff7b)
- [Issue 4](https://github.com/Saaras859/Team-Influencer-Innovators/issues/4)


