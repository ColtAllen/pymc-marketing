```mermaid
---
title: deserialize.py
---
classDiagram
    class Deserializer {
        + IsType is_type
        + Deserialize deserialize
    }

    class DeserializableError {
        - __init__(self, data) None
    }

    DeserializableError --|> Exception
```
