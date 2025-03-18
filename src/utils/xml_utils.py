import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class XMLEventExtractor:
    """
    A utility class for extracting event data from XSEED XML files.
    Focuses on extracting event success labels and clip filenames for ML training.
    """
    
    def __init__(self, xml_content: str):
        """
        Initialize the event extractor with XML content.
        
        Args:
            xml_content: XML content as string
        """
        self.xml_content = xml_content
        self.root = ET.fromstring(xml_content)
    
    def extract_events(self) -> List[Dict]:
        """
        Extract all relevant events from the XML file.
        
        Returns:
            List of event dictionaries with relevant fields for ML training
        """
        events = []
        instances = self.root.find('ALL_INSTANCES')
        
        if instances is None:
            return events
        
        for instance in instances.findall('instance'):
            # Only process impact events that have success/failure labels
            label = instance.findtext('label', '')
            group = instance.findtext('group', '')
            
            # Focus on events that have a success/failure outcome
            if label == 'Impact' and group in ['Pass', 'Cross', 'Long Ball', 'Shot']:
                event = self._extract_event_data(instance)
                if event:  # Only add if we have success/failure data
                    events.append(event)
        
        return events
    
    def _extract_event_data(self, instance: ET.Element) -> Optional[Dict]:
        """
        Extract relevant data from an event instance.
        
        Args:
            instance: XML element representing an event instance
            
        Returns:
            Dictionary with event data, or None if success field missing
        """
        event = {
            'id': instance.findtext('ID', ''),
            'start': float(instance.findtext('start', '0')),
            'end': float(instance.findtext('end', '0')),
            'code': instance.findtext('code', ''),  # Player name
            'label': instance.findtext('label', ''),
            'group': instance.findtext('group', ''),
            'team': instance.findtext('team', ''),
            'local_time': instance.findtext('local_time', ''),
        }
        
        # Format GPS timestamp for filename
        if event['local_time']:
            try:
                dt = datetime.strptime(event['local_time'], '%Y-%m-%d %H:%M:%S')
                event['formatted_time'] = dt.strftime('%Y%m%d_%H%M%S')
            except ValueError:
                event['formatted_time'] = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            event['formatted_time'] = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Extract success/failure label from metadata
        metadata_inst = instance.find('metadata')
        if metadata_inst is not None:
            event['metadata'] = {}
            for child in metadata_inst:
                event['metadata'][child.tag] = child.text
            
            # Check for success/failure field - crucial for our ML task
            if 'successful' in event['metadata']:
                # Convert to binary label (1 for success, 0 for failure)
                event['success'] = 1 if event['metadata']['successful'] == 'True' else 0
                return event
        
        # If we didn't find success field, return None
        return None
    
    def generate_clip_filename(self, event: Dict) -> str:
        """
        Generate a standardized filename for video clips.
        
        Args:
            event: Event dictionary containing event details (must include user_id)
            
        Returns:
            str: Formatted filename for the clip
        """
        # Ensure user_id is present in the event
        if 'user_id' not in event:
            raise ValueError("Event must contain user_id before generating clip filename")
            
        # Extract the known name (player name)
        known_name = event['code'].replace('.', '_').replace(' ', '')
        user_id = event['user_id']

        # Transform from start (sec) to start in min_sec
        event_time = (event['start'] + event['end']) / 2
        event_time = str(int(event_time // 60)).zfill(2) + str(int(event_time % 60)).zfill(2)
        
        # Extract event type and timestamp
        group = event['group']
        timestamp = event['formatted_time']
        
        # Create standardized filename:
        # format: {known_name}_{group}_{event_time}_{user_id}_{timestamp}.mp4
        filename = f"{known_name}_{group}_{event_time}_{user_id}_{timestamp}.mp4"
        
        # Replace any problematic characters
        filename = filename.replace(' ', '').replace('/', '_').replace('\\', '_')
        
        return filename
    
    def extract_events_with_filenames(self, player_mapping: Dict[str, str]) -> List[Dict]:
        """
        Extract events with their clip filenames.
        
        Args:
            player_mapping: Dictionary mapping player names to user IDs
            
        Returns:
            List of dictionaries containing event data and clip filenames
        """
        events = self.extract_events()
        
        for event in events:
            # First, apply user_id from player mapping
            code = event['code']
            if code in player_mapping:
                event['user_id'] = player_mapping[code]
            else:
                # Use a default user_id for players not in the mapping
                event['user_id'] = 'unknown'
            
            # Now generate the clip filename with the user_id included
            event['clip_filename'] = self.generate_clip_filename(event)
            
            # Extract recipient player info if available
            if 'to_player_name' in event.get('metadata', {}):
                event['to_player'] = event['metadata']['to_player_name']
        
        return events
    
    def get_player_mapping(self) -> Dict[str, str]:
        """
        Extract player mapping from metadata section (name to ID).
        
        Returns:
            Dictionary mapping player known names to their IDs
        """
        player_mapping = {}
        metadata_elem = self.root.find('METADATA')
        
        if metadata_elem is None:
            return player_mapping
        
        # Process home team
        home_team = metadata_elem.find('homeTeam')
        if home_team is not None:
            for player in home_team.findall('.//player'):
                known_name = player.findtext('knownName', '')
                player_id = player.findtext('id', '')
                if known_name and player_id:
                    player_mapping[known_name] = player_id
        
        # Process away team
        away_team = metadata_elem.find('awayTeam')
        if away_team is not None:
            for player in away_team.findall('.//player'):
                known_name = player.findtext('knownName', '')
                player_id = player.findtext('id', '')
                if known_name and player_id:
                    player_mapping[known_name] = player_id
        
        return player_mapping