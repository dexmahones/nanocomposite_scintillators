import os
from bs4 import BeautifulSoup
import re

with open("chem_props.txt",'r') as infile:
    html_content = infile.read()

    # Parse the HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # Target all the individual table rows
    rows = soup.find_all("div", class_="css-gdvca5-row")

    # Build the Markdown table headers
    markdown_table = "| Property | Value |\n| --- | --- |\n"

    for row in rows:
        # Extract the label/property name
        label_element = row.find("h3", class_="css-ly88zs-label")
        # Extract the corresponding value
        value_element = row.find("span", class_="css-ggy4-value")

        if label_element and value_element:
            # Get text, strip white space, and fix HTML spaces like &nbsp;
            label = label_element.get_text(strip=True)
            value = value_element.get_text(strip=True)
            value = re.sub(r"\u00a0", " ", value)  # Cleans up &nbsp; artifacts

            # Add the row to our Markdown table string
            markdown_table += f"| {label} | {value} |\n"

    print(markdown_table)