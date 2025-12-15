# src/crew.py - Updated with proper resume generation

import os
import yaml
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, FileReadTool, DirectoryReadTool
from src.tools.file_tools import ResumeFileProcessor, ResumeGenerator, EnhancedResumeContentGenerator
from src.tools.analysis_tools import ResumeAnalyzer
from src.tools.web_tools import JobSearchTool
from src.knowledge.ai_ds_requirements import AI_DS_REQUIREMENTS

# Load environment variables
load_dotenv()

class ResumeAnalyzerCrew:
    def __init__(self):
        self.agents_config = self.load_config('config/agents.yaml')
        self.tasks_config = self.load_config('config/tasks.yaml')
        self.llm = self.setup_llm()
        self.setup_tools()
        self.setup_agents()
        
    def load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def setup_llm(self):
        """Configure LLM for CrewAI with fallback options"""
        try:
            # Try Gemini first
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if gemini_api_key:
                llm = LLM(
                    model="gemini/gemini-2.0-flash",
                    api_key=gemini_api_key,
                    temperature=0.1,
                    max_tokens=4000
                )
                print("✅ Gemini LLM configured successfully")
                return llm
            
            # Fallback to OpenAI if available
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                llm = LLM(
                    model="gpt-3.5-turbo",
                    api_key=openai_api_key,
                    temperature=0.1,
                    max_tokens=4000
                )
                print("✅ OpenAI LLM configured successfully")
                return llm
            
            # If no API keys, use default (this might not work in production)
            print("⚠️ No API keys found, using default LLM")
            return None
            
        except Exception as e:
            print(f"❌ Error configuring LLM: {e}")
            return None
    
    def setup_tools(self):
        """Initialize all tools"""
        self.file_processor = ResumeFileProcessor()
        self.resume_analyzer = ResumeAnalyzer()
        self.resume_generator = ResumeGenerator()
        self.job_search_tool = JobSearchTool()
        self.file_read_tool = FileReadTool()
    
    def setup_agents(self):
        """Create agents based on YAML configuration"""
        
        # Resume Analyzer Agent
        self.resume_analyzer_agent = Agent(
            role=self.agents_config['resume_analyzer']['role'],
            goal=self.agents_config['resume_analyzer']['goal'],
            backstory=self.agents_config['resume_analyzer']['backstory'],
            verbose=self.agents_config['resume_analyzer']['verbose'],
            allow_delegation=self.agents_config['resume_analyzer']['allow_delegation'],
            max_iter=self.agents_config['resume_analyzer']['max_iter'],
            memory=self.agents_config['resume_analyzer']['memory'],
            tools=[self.file_read_tool, self.resume_analyzer],
            llm=self.llm
        )
        
        # Resume Enhancer Agent  
        self.resume_enhancer_agent = Agent(
            role=self.agents_config['resume_enhancer']['role'],
            goal=self.agents_config['resume_enhancer']['goal'],
            backstory=self.agents_config['resume_enhancer']['backstory'],
            verbose=self.agents_config['resume_enhancer']['verbose'],
            allow_delegation=self.agents_config['resume_enhancer']['allow_delegation'],
            max_iter=self.agents_config['resume_enhancer']['max_iter'],
            memory=self.agents_config['resume_enhancer']['memory'],
            tools=[self.resume_analyzer],
            llm=self.llm
        )
        
        # Resume Generator Agent
        self.resume_generator_agent = Agent(
            role=self.agents_config['resume_generator']['role'],
            goal=self.agents_config['resume_generator']['goal'], 
            backstory=self.agents_config['resume_generator']['backstory'],
            verbose=self.agents_config['resume_generator']['verbose'],
            allow_delegation=self.agents_config['resume_generator']['allow_delegation'],
            max_iter=self.agents_config['resume_generator']['max_iter'],
            memory=self.agents_config['resume_generator']['memory'],
            tools=[self.file_read_tool, self.resume_generator],
            llm=self.llm
        )
        
        # Web Researcher Agent
        self.web_researcher_agent = Agent(
            role=self.agents_config['web_researcher']['role'],
            goal=self.agents_config['web_researcher']['goal'],
            backstory=self.agents_config['web_researcher']['backstory'],
            verbose=self.agents_config['web_researcher']['verbose'], 
            allow_delegation=self.agents_config['web_researcher']['allow_delegation'],
            max_iter=self.agents_config['web_researcher']['max_iter'],
            memory=self.agents_config['web_researcher']['memory'],
            tools=[self.job_search_tool],
            llm=self.llm
        )
    
    async def analyze_with_static_knowledge(self, file_path: str):
        """Analyze resume using static knowledge base"""
        
        try:
            # Create tasks with better error handling
            analyze_task = Task(
                description=self.tasks_config['analyze_resume']['description'].format(file_path=file_path),
                expected_output=self.tasks_config['analyze_resume']['expected_output'],
                agent=self.resume_analyzer_agent
            )
            
            enhance_task = Task(
                description=self.tasks_config['enhance_resume']['description'],
                expected_output=self.tasks_config['enhance_resume']['expected_output'],
                agent=self.resume_enhancer_agent,
                context=[analyze_task]
            )
            
            # Add static knowledge context
            static_context = f"Use this AI/DS job requirements knowledge base: {AI_DS_REQUIREMENTS}"
            analyze_task.description += f"\n\nKnowledge Base: {static_context}"
            
            # Create crew with error handling
            crew = Crew(
                agents=[self.resume_analyzer_agent, self.resume_enhancer_agent],
                tasks=[analyze_task, enhance_task],
                process=Process.sequential,
                verbose=True,
                max_rpm=10,  # Add rate limiting
                memory=False  # Disable memory to avoid issues
            )
            
            # Execute crew
            result = crew.kickoff()
            return self.parse_analysis_result(result)
            
        except Exception as e:
            print(f"❌ Error in analyze_with_static_knowledge: {e}")
            # Return a fallback analysis using direct tool call
            return self.fallback_analysis(file_path)
    
    async def analyze_with_web_search(self, file_path: str):
        """Analyze resume using web search for current job requirements"""
        
        try:
            # Create tasks
            research_task = Task(
                description=self.tasks_config['research_jobs']['description'],
                expected_output=self.tasks_config['research_jobs']['expected_output'],
                agent=self.web_researcher_agent
            )
            
            analyze_task = Task(
                description=self.tasks_config['analyze_resume']['description'].format(file_path=file_path),
                expected_output=self.tasks_config['analyze_resume']['expected_output'],
                agent=self.resume_analyzer_agent,
                context=[research_task]
            )
            
            enhance_task = Task(
                description=self.tasks_config['enhance_resume']['description'],
                expected_output=self.tasks_config['enhance_resume']['expected_output'],
                agent=self.resume_enhancer_agent,
                context=[research_task, analyze_task]
            )
            
            # Create crew
            crew = Crew(
                agents=[self.web_researcher_agent, self.resume_analyzer_agent, self.resume_enhancer_agent],
                tasks=[research_task, analyze_task, enhance_task],
                process=Process.sequential,
                verbose=True,
                max_rpm=10,
                memory=False
            )
            
            # Execute crew
            result = crew.kickoff()
            return self.parse_analysis_result(result)
            
        except Exception as e:
            print(f"❌ Error in analyze_with_web_search: {e}")
            # Fall back to static analysis
            return await self.analyze_with_static_knowledge(file_path)
    
    def fallback_analysis(self, file_path: str):
        """Fallback analysis using direct tool calls when crew fails"""
        try:
            # Read file directly
            resume_content = ""
            if file_path.endswith('.pdf'):
                resume_content = self.file_processor._extract_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                resume_content = self.file_processor._extract_from_docx(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    resume_content = f.read()
            
            # Use resume analyzer tool directly
            analysis_result = self.resume_analyzer._run(
                resume_text=resume_content,
                job_requirements=str(AI_DS_REQUIREMENTS)
            )
            
            # Parse the direct tool result
            return {
                "strengths": self._extract_from_analysis(analysis_result, "strengths"),
                "weaknesses": self._extract_from_analysis(analysis_result, "weaknesses"),
                "ats_score": self._extract_ats_score_from_analysis(analysis_result),
                "missing_keywords": self._extract_keywords_from_analysis(analysis_result),
                "improvements": self._extract_from_analysis(analysis_result, "improvement"),
                "raw_result": analysis_result,
                "original_content": resume_content
            }
            
        except Exception as e:
            print(f"❌ Fallback analysis failed: {e}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "strengths": [],
                "weaknesses": ["Could not analyze resume"],
                "ats_score": 0,
                "missing_keywords": [],
                "improvements": [],
                "raw_result": "",
                "original_content": ""
            }
    
    def _extract_from_analysis(self, analysis_text: str, section: str) -> list:
        """Extract specific section from analysis text"""
        try:
            import re
            # Find section in text
            pattern = rf'{section}.*?:\s*(.*?)(?:\n\n|\n[A-Z]|$)'
            match = re.search(pattern, analysis_text, re.IGNORECASE | re.DOTALL)
            
            if match:
                content = match.group(1)
                # Extract bullet points
                bullets = re.findall(r'[•\-\*]\s*([^\n]+)', content)
                return bullets[:7] if bullets else [content.strip()[:100]]
            
            return []
        except:
            return []
    
    def _extract_ats_score_from_analysis(self, analysis_text: str) -> int:
        """Extract ATS score from analysis text"""
        try:
            import re
            score_match = re.search(r'ats.*?(?:score|compatibility).*?(\d+)', analysis_text.lower())
            return int(score_match.group(1)) if score_match else 6
        except:
            return 6
    
    def _extract_keywords_from_analysis(self, analysis_text: str) -> list:
        """Extract missing keywords from analysis text"""
        try:
            import re
            # Look for missing keywords section
            keyword_match = re.search(r'missing.*?keywords?:?\s*(.*?)(?:\n\n|\n[A-Z]|$)', analysis_text.lower(), re.DOTALL)
            if keyword_match:
                keywords_text = keyword_match.group(1)
                # Extract individual keywords
                keywords = re.findall(r'[a-zA-Z+\-\.]+', keywords_text)
                return [k for k in keywords if len(k) > 2][:10]
            return []
        except:
            return []
    
    async def generate_enhanced_resume(self, file_path: str, analysis_data: dict):
        """Generate enhanced resume based on analysis"""
        
        try:
            # Get original content
            original_content = analysis_data.get('original_content', '')
            
            if not original_content:
                # Extract original content if not available
                if file_path.endswith('.pdf'):
                    original_content = self.file_processor._extract_from_pdf(file_path)
                elif file_path.endswith('.docx'):
                    original_content = self.file_processor._extract_from_docx(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
            
            # Generate enhanced content using the content generator
            improvements = analysis_data.get('improvements', analysis_data.get('weaknesses', []))
            enhanced_content = EnhancedResumeContentGenerator.enhance_resume_content(
                original_content, 
                analysis_data, 
                improvements
            )
            
            # Create output path
            filename_base = os.path.splitext(os.path.basename(file_path))[0]
            output_path = f"outputs/{filename_base}_enhanced.docx"
            
            # Ensure outputs directory exists
            os.makedirs("outputs", exist_ok=True)
            
            # Generate the actual DOCX file
            self.resume_generator.generate_docx(enhanced_content, output_path)
            
            print(f"✅ Enhanced resume generated: {output_path}")
            return f"{filename_base}_enhanced.docx"
            
        except Exception as e:
            print(f"❌ Error generating enhanced resume: {e}")
            
            # Fallback: Create a simple enhanced resume
            try:
                filename_base = os.path.splitext(os.path.basename(file_path))[0]
                output_path = f"outputs/{filename_base}_enhanced.docx"
                
                # Use original content or create sample content
                content_to_use = analysis_data.get('original_content', '')
                if not content_to_use or len(content_to_use.strip()) < 50:
                    content_to_use = self.resume_generator._generate_sample_resume()
                
                # Generate with fallback content
                self.resume_generator.generate_docx(content_to_use, output_path)
                
                return f"{filename_base}_enhanced.docx"
                
            except Exception as fallback_error:
                print(f"❌ Fallback generation also failed: {fallback_error}")
                return None
    
    def parse_analysis_result(self, result):
        """Parse crew result into structured format"""
        try:
            # Extract key information from result
            result_text = str(result)
            
            # Simple parsing - in production, use more sophisticated extraction
            analysis = {
                "strengths": self.extract_section(result_text, "strengths"),
                "weaknesses": self.extract_section(result_text, "weaknesses"), 
                "ats_score": self.extract_ats_score(result_text),
                "missing_keywords": self.extract_keywords(result_text),
                "improvements": self.extract_section(result_text, "improvement"),
                "raw_result": result_text
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error parsing result: {e}")
            return {
                "error": str(e),
                "raw_result": str(result),
                "strengths": [],
                "weaknesses": [],
                "ats_score": 0,
                "missing_keywords": [],
                "improvements": []
            }
    
    def extract_section(self, text: str, section: str) -> list:
        """Extract bullet points from a section"""
        try:
            lines = text.lower().split('\n')
            section_lines = []
            in_section = False
            
            for line in lines:
                if section in line:
                    in_section = True
                    continue
                if in_section and line.strip().startswith(('-', '•', '*')):
                    section_lines.append(line.strip())
                elif in_section and not line.strip():
                    continue
                elif in_section and line.strip() and not line.strip().startswith(('-', '•', '*')):
                    break
                    
            return section_lines[:7]  # Limit to 7 items
        except:
            return []
    
    def extract_ats_score(self, text: str) -> int:
        """Extract ATS compatibility score"""
        try:
            import re
            score_match = re.search(r'ats.*?(\d+)', text.lower())
            return int(score_match.group(1)) if score_match else 6
        except:
            return 6
    
    def extract_keywords(self, text: str) -> list:
        """Extract missing keywords"""
        try:
            import re
            keyword_section = re.search(r'missing.*?keywords?:?(.*?)(?:\n\n|\n[A-Z]|$)', text.lower(), re.DOTALL)
            if keyword_section:
                keywords = re.findall(r'[a-zA-Z+\-\.]+', keyword_section.group(1))
                return [k for k in keywords if len(k) > 2][:10]
            return []
        except:
            return []