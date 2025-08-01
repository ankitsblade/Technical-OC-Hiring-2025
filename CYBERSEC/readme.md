# Cybersecurity Head
## Your task is to design and implement a secure file storage system that lets users upload/download files with encryption.

### Success Criterion

- **Implement Core Cryptographic Primitives**: Begin by building the fundamental cryptographic algorithms from scratch (or demonstrating a ground-up understanding). This includes implementing AES for symmetric file encryption, RSA for asymmetric key management, PBKDF2 for password hashing, and SHA-256 for creating data fingerprints.

- **Design and Build a Secure Login System**: Use your implemented PBKDF2 primitive and a salted hashing scheme to create a secure user registration and login system. This system must securely store user credentials in a database and manage authenticated sessions.

- **Develop the Encrypted File Storage Core**: Create the main functionality for secure file uploads and downloads. Implement a hybrid encryption scheme where files are encrypted with a unique AES key, and that AES key is in turn encrypted with the user's public RSA key before being stored.

- **Implement Integrity Checks and Key Management**: Ensure data hasn't been tampered with by implementing an integrity check system, using your SHA-256 implementation to create a hash or HMAC of all files. Additionally, build a simple interface for users to generate and manage their RSA key pairs.

- **Simulate and Patch a Security Vulnerability**: Act as both an attacker and a defender. Identify a potential vulnerability in your system (e.g., allowing unauthorized file access, mishandling cryptographic keys), create a proof-of-concept attack to exploit it, and then patch the vulnerability to secure the system.

- **Create the Demo Video with Security Post-Mortem**: Produce a short video **(<10 min)** that demonstrates the system's functionality. The video must showcase the simulated attack and the defense, and include a brief security report covering the threat model, the vulnerability post-mortem, and design choices.

