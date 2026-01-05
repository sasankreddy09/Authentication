# Dashboard Application

## Next.js + Supabase Authentication Dashboard

A modern dashboard application built using **Next.js App Router** and **Supabase Authentication**, focused on clean UI, secure auth handling, user personalization, and client-side caching.

---

## Overview

This project implements a simple dashboard that authenticates users via Supabase, displays a personalized greeting, and allows users to securely log out.  
It follows real-world development practices such as defensive programming, modular components, and predictable authentication flow.

---

## Features

- Secure authentication using Supabase
- Personalized user greeting
- Safe access to user metadata
- Fallback name handling
- Client-side caching with localStorage
- Logout functionality with redirect
- Clean and responsive UI using CSS Modules

---

## Tech Stack

- **Frontend:** Next.js (App Router), React
- **Authentication:** Supabase Auth
- **Styling:** CSS Modules
- **Routing:** Next.js Navigation
- **State Management:** React Hooks
- **Storage:** Browser localStorage

---
---

## Setup & Installation

### Clone the repository
```bash
git clone <repository-url>
cd dashboard-app
Install dependencies
bash
Copy code
npm install
Run the development server
bash
Copy code
npm run dev
Environment Variables
Create a .env.local file in the root directory:

env
Copy code
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

