---
toc: True
comments: True
layout: post
title: My struggle with coordinates
type: tangibles
courses: {'csp': {'week': 10}}
---

# My Struggle with Coordinates: A Journey to Understanding Scaling and Coordinate Systems

In the world of coding, every programmer encounters challenges that test their problem-solving skills. One of my recent struggles involved dealing with coordinates in a graphical project. At first, it seemed straightforward, but as I delved deeper, I realized that the image was scaled improperly, leading to confusion in my coordinate system. Let me share my journey of overcoming this obstacle and altering the coordinate system to achieve the desired results.

## The Initial Confusion

I started working on a graphical project that involved plotting points on an image. I assumed that the coordinates in the image file directly corresponded to the pixels on my screen. However, when I plotted points using these coordinates, they didn't align as expected. Frustration set in as I couldn't understand why my points were appearing in seemingly random positions on the image.

## Realizing the Scaling Issue

After some research and experimentation, I realized that the image I was working with had been scaled improperly. The dimensions of the image in pixels did not match the dimensions that the program was assuming. This mismatch in scaling was the root cause of my coordinate woes. Understanding this, I knew I had to find a way to rectify the situation.

## Altering the Coordinate System

To address the scaling issue, I needed to alter the coordinate system used in my program. Instead of relying on the raw pixel coordinates from the image, I decided to implement a scaling factor. By multiplying the original coordinates by this factor, I could map them correctly onto the scaled image. This adjustment was crucial to ensuring that my plotted points would align perfectly with the image, regardless of its dimensions.

## The Eureka Moment

Implementing the scaling factor was the turning point in my struggle. When I ran the program again, the points I plotted were finally in the right positions on the image. It was a eureka moment, and the sense of accomplishment was incredibly satisfying.

In conclusion, this experience taught me the importance of understanding the intricacies of the coordinate system and the impact of scaling on graphical projects. Through perseverance, research, and code adjustments, I was able to overcome the challenges and achieve the desired outcome. This journey not only enhanced my coding skills but also deepened my understanding of graphical concepts—a valuable lesson that will undoubtedly benefit me in future projects.

## Issues

- [Issue #1](https://github.com/will-w-cheng/team-influencer-innovator-backend/issues/1)

## Commits

- [Commit where you started](https://github.com/will-w-cheng/team-influencer-innovator-backend/commit/bc8e899d55c9c6aca4f8cd313add2ce64cea5bc0)
- [Recent Commit](https://github.com/will-w-cheng/team-influencer-innovator-backend/commit/babf0bbba02cbc02967ed0caeeba0e4fd8b5a533)



