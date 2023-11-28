---
toc: True
comments: True
layout: post
title: InternetLesson
type: tangibles
courses: {'csp': {'week': 15}}
---

## Internet History Fun Facts
- A catalyst in the formation of the Internet was the heating up of the Cold War. The Soviet Union's launch of the Sputnik satellite spurred the U.S. Defense Department to consider ways information could still be disseminated even after a nuclear attack.
- ARPANET was created in 1969 and provided the foundation for what came to be known as the Internet
- January 1, 1983 is the official birthday of the modern internet

## FOR ME LATER MAKE SURE TO ADD DISCORD IMAGE OF 
https://files.slack.com/files-tmb/TUDAF53UJ-F065Y6QNP8E-1ab397c1fe/image_360.png

## OSI Model

The OSI (Open Systems Interconnection) model is a conceptual framework that standardizes the functions of a telecommunication or computing system into seven abstraction layers. Each layer in the OSI model performs specific functions, and together they facilitate communication between different systems. Here's a brief overview of what each layer does:  

**1. Physical Layer (Layer 1):**
- Manages the physical connection and transmission of raw data bits.

**2. Data Link Layer (Layer 2):**
- Ensures reliable communication over a physical layer, handling framing and addressing.

**3. Network Layer (Layer 3):**
- Focuses on logical addressing and routing data between different networks.

**4. Transport Layer (Layer 4):**
- Manages end-to-end communication, ensuring reliable and ordered data delivery.



## OSI Model in a network

**1. Physical Layer (Layer 1):**
![Layer 1](https://files.slack.com/files-pri/TUDAF53UJ-F067AG02E14/layers-of-the-osi-model-illustrated-818017-finalv1-2-ct-ed94d33e885a41748071ca15289605c9.png)


**1. Physical Layer (Layer 1) Example:**
- **Scenario:** Transmitting data over an Ethernet cable.
- **Functionality:**
  - Manages the physical connection between devices.
  - Handles the transmission of raw data bits over the Ethernet cable.
  - Deals with aspects like cable types, connectors, and signal modulation.


**Data Link Layer (Layer 2) Example:**
- **Scenario:** Communication between two computers over an Ethernet network.
- **Functionality:**
  - Breaks data into frames with start and stop bits for accurate transmission.
  - Manages MAC addresses for unique identification of devices.
  - Regulates frame flow to prevent network congestion.
  - Implements error-checking mechanisms like checksums or CRC for data integrity.


**Network Layer (Layer 3) Example:**
- **Scenario:** Routing data between two networks.
- **Functionality:**
  - Focuses on logical addressing to uniquely identify devices on different networks.
  - Determines optimal paths for data to travel from the source to the destination.
  - Manages communication between networks using routers.
  - Provides a foundation for end-to-end communication across interconnected networks.


**Transport Layer (Layer 4) Example:**
- **Scenario:** Ensuring reliable communication between two applications.
- **Functionality:**
  - Manages end-to-end communication, ensuring data arrives reliably and in the correct order.
  - Provides error detection and recovery mechanisms for data integrity.
  - Uses protocols like TCP for connection-oriented and reliable communication.
  - Supports protocols like UDP for connectionless and faster communication when some data loss is acceptable.


**Transport Layer (Layer 4) - TCP vs. UDP**

### TCP (Transmission Control Protocol)
- **Scenario:** Reliable communication requiring guaranteed data delivery.
- **Functionality:**
  - Establishes a connection between sender and receiver before data transfer.
  - Ensures reliable and ordered delivery of data.
  - Implements error detection and correction mechanisms.
  - Well-suited for applications where data integrity is crucial, such as file transfers and web browsing.

### UDP (User Datagram Protocol)
- **Scenario:** Faster communication with acceptable data loss.
- **Functionality:**
  - Connectionless protocol without establishing a connection before data transfer.
  - Does not guarantee reliable or ordered delivery of data.
  - Suitable for real-time applications like video streaming and online gaming, where speed is prioritized over perfect data transmission.
  - Minimal overhead, making it more lightweight than TCP.

**TCP and UDP are two different approaches to data transport, each suited to specific application requirements. The choice between them depends on the priorities of the application: reliability (TCP) or speed (UDP).**




