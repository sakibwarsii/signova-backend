from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY

def create_pdf(filename, title, content):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    
    story = []
    
    # Title
    story.append(Paragraph(title, styles["Heading1"]))
    story.append(Spacer(1, 12))
    
    # Content
    for paragraph in content:
        story.append(Paragraph(paragraph, styles["Justify"]))
        story.append(Spacer(1, 12))
        
    doc.build(story)
    print(f"Created {filename}")

def main():
    # EVS Content
    evs_title = "Class 3 EVS: Plants and Trees"
    evs_content = [
        "Plants are living things that grow in the earth and have a stem, leaves, and roots.",
        "There are many different types of plants. Trees are very large plants. They have a thick, woody stem called a trunk.",
        "Shrubs are smaller than trees and have many thin, woody stems. Herbs are small plants with soft, green stems.",
        "Plants need water, sunlight, and air to grow. They make their own food using sunlight. This process is called photosynthesis.",
        "Roots hold the plant in the soil and absorb water. The stem carries water to the leaves. The leaves make food for the plant."
    ]
    create_pdf("Sample_EVS_Class3.pdf", evs_title, evs_content)

    # Maths Content
    maths_title = "Class 2 Maths: Addition and Subtraction"
    maths_content = [
        "Addition means putting things together. When we add two numbers, we get their sum.",
        "For example, if you have 3 apples and your friend gives you 2 more apples, you have 5 apples in total. 3 + 2 = 5.",
        "Subtraction means taking things away. When we subtract one number from another, we find the difference.",
        "For example, if you have 5 candies and you eat 2 candies, you are left with 3 candies. 5 - 2 = 3.",
        "Addition and subtraction are inverse operations. If 3 + 2 = 5, then 5 - 2 = 3."
    ]
    create_pdf("Sample_Maths_Class2.pdf", maths_title, maths_content)

    # English Content
    english_title = "Class 4 English: The Thirsty Crow"
    english_content = [
        "One hot summer day, a thirsty crow was looking for water. He flew all over the fields but could not find any water.",
        "After a long time, he saw a pitcher under a tree. He flew straight down to see if there was any water inside.",
        "Yes, he could see some water inside the pitcher! But the water level was very low. The crow tried to push his head into the pitcher, but the neck of the pitcher was too narrow.",
        "He looked around and saw some pebbles on the ground. He got an idea. He picked up the pebbles one by one and dropped them into the pitcher.",
        "As he dropped more and more pebbles, the water came up to the top. The crow drank the water and flew away happily."
    ]
    create_pdf("Sample_English_Class4.pdf", english_title, english_content)

if __name__ == "__main__":
    main()
