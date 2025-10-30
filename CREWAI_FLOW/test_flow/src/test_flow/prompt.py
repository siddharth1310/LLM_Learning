lecture_creation_user_prompt = """
            You are an expert educator tasked with preparing a very detailed and comprehensive lecture on the topic: "${topic}".
            This lecture will be delivered by ${name} to an audience with expertise level: ${audience_level}.

            Your output must include three distinct parts, formatted clearly using Markdown:

            1. **Personalized Greeting Message:**  
                - A warm, engaging introduction from ${name} to connect with the audience.

            2. **Important Considerations:**  
                - A bulleted list of key points and things to keep in mind specific to the audience level to effectively engage them.

            3. **Detailed Lecture Content:**  
                - Compose a thorough lecture that covers every crucial aspect of the topic.  
                - Begin with an attention-grabbing introduction to hook the audience.  
                - The lecture should be informative, well-organized, and use language suitable for the specified audience expertise.

            Additional guidelines:  
                - Use Markdown formatting extensively — including headings, lists, emphasis (bold, italics), and paragraphs — to enhance readability.  
                - Ensure clarity and professionalism in tone while keeping the content engaging.  
                - Tailor explanations to the audience's knowledge level — for beginners, focus on foundational concepts; for advanced audiences, include in-depth analysis and technical details.  
                - Return your complete response as a valid JSON object with keys:  
                - "greeting_message"  
                - "important_points" (list of strings)  
                - "lecture_content"

            Do not include any additional text outside this JSON structure.
            """

lecture_evaluator_prompt = """
            You are a critically acclaimed writer and linguistic expert tasked with evaluating the following lecture.
            
            Evaluate the lecture **strictly and exclusively** against the exact criteria listed below.  
            Do **not** invent, add, remove, or rename any criteria.  
            
            For each criterion, assign an integer score from **1 (very poor)** to **10 (excellent)** based solely on the lecture's content.  
            
            Your response must be a **valid JSON object** with keys exactly matching the criteria names below and integer scores as values.  
            Do **not** include any explanations, commentary, or formatting outside this JSON.  
            
            ### Important Points to Evaluate
            ${important_points}
            
            ## Lecture to evaluate:
            ${lecture_content}
            """
